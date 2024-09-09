#!/bin/bash
#
if [[ $# -lt 6 ]] ; then
    echo "Usage: $0 <source corpus> <translation corpus> <reference corpus> <output directory> <number of CPUs> <locale of target language> [optional: <alignment dir>]"
    exit 1
fi

#
# Get the arguments passed to this script
#

source_corpus=`realpath ${1}`
translation_corpus=`realpath ${2}`
reference_corpus=`realpath ${3}`
output_directory=`realpath ${4}`
nb_cpus=${5}
tl_locale=${6}
align_dir=${7}

#
# Set a few useful paths
#

work_dir=`dirname -- ${0}`
work_dir=`realpath ${work_dir}`
scripts_dir=${work_dir}/scripts
tmp_dir=${output_directory}/tmp
mkdir -p ${tmp_dir}
cache_dir=${output_directory}/cache
mkdir -p ${cache_dir}

#
# Check if alignment dir was provided and exists
#

already_alignments=0
if [ ! -z ${align_dir} ] && [ -d ${align_dir} ] && \
    [ -f ${align_dir}/a.s2t.params ] && [ -f ${align_dir}/align.s2t ] && [ -f ${align_dir}/align.s2t.err ] && \
    [ -f ${align_dir}/a.t2s.params ] && [ -f ${align_dir}/align.t2s ] && [ -f ${align_dir}/align.t2s.err ]
then
    already_alignments=1
    echo "Alignments are found in ${align_dir}"
else
    align_dir=${output_directory}/alignments
    mkdir -p ${align_dir}
fi

#
# Set the cache directories for HuggingFace Transformers and Datasets
#

export HF_DATASETS_CACHE=${cache_dir}
export HF_HOME=${cache_dir}

#
# Tokenize corpora using the XLM-R tokenizer from Huggingface
#

echo "Tokenizing data."
python ${scripts_dir}/mtqe_tokenizer.py -s ${source_corpus} -t ${translation_corpus} -r ${reference_corpus} -o ${tmp_dir}/preprocess

#
# Compute MT sentence-level scores given the translation and reference corpora
#

echo "Computing sentence-level scores."
mkdir -p ${tmp_dir}/split
paste ${tmp_dir}/preprocess.translation.tok ${tmp_dir}/preprocess.reference.tok > ${tmp_dir}/preprocess.translation_reference.tok.para
split -d -n l/${nb_cpus} ${tmp_dir}/preprocess.translation_reference.tok.para ${tmp_dir}/split/para.
for i in ${tmp_dir}/split/para.*; do
    f=`echo $i | rev | cut -d"." -f1 | rev` ;
    cut -f1 ${i} > ${tmp_dir}/split/tra.${f}
    cut -f2 ${i} > ${tmp_dir}/split/ref.${f}
    python ${scripts_dir}/compute_sacrebleu_metrics.py \
        -t ${tmp_dir}/split/tra.${f} -r ${tmp_dir}/split/ref.${f} -o ${tmp_dir}/split/score.${f} & 
done
wait
for j in bleu chrf; do for i in ${tmp_dir}/split/score.*.${j}; do cat $i; echo; done > ${output_directory}/translation.${j}; done
rm -rf ${tmp_dir}/split
rm ${tmp_dir}/preprocess.translation_reference.tok.para

#
# Source--translation token alignments from previously trained fast_align model
#

echo "Computing source--reference token alignments using fast_align."

# Create single file by pasting source and translation parallel corpora

# python ${scripts_dir}/make_parallel_file.py ${tmp_dir}/preprocess.source.tok ${tmp_dir}/preprocess.translation.tok \
#     ${tmp_dir}/source_translation_parallel_corpus

# Create single file by pasting source and reference parallel corpora

python ${scripts_dir}/make_parallel_file.py ${tmp_dir}/preprocess.source.tok ${tmp_dir}/preprocess.reference.tok \
    ${tmp_dir}/source_reference_parallel_corpus

# Train alignment models in both directions (source to target and target to source)

if [ $already_alignments == 1 ]; then
    echo "Using given alignments (in ${align_dir})"
elif [ $already_alignments == 0 ]; then
    OMP_NUM_THREADS=${nb_cpus} ${scripts_dir}/fast_align/build/fast_align -i ${tmp_dir}/source_reference_parallel_corpus \
        -d -o -v -p ${align_dir}/a.s2t.params > ${align_dir}/align.s2t 2> ${align_dir}/align.s2t.err
    OMP_NUM_THREADS=${nb_cpus} ${scripts_dir}/fast_align/build/fast_align -i ${tmp_dir}/source_reference_parallel_corpus \
        -d -o -v -r -p ${align_dir}/a.t2s.params > ${align_dir}/align.t2s 2> ${align_dir}/align.t2s.err
else
    echo "Something wrong happened."
    exit
fi

# Split source--translation parallel file and get alignments

# python ${scripts_dir}/force_align.py -c ${tmp_dir}/source_translation_parallel_corpus -fa ${scripts_dir}/fast_align/build \
#     -fp ${align_dir}/align.s2t -fe ${align_dir}/align.s2t.err -rp ${align_dir}/align.t2s -re ${align_dir}/align.t2s.err \
#     -he grow-diag-final-and -o ${tmp_dir}/source_translation_alignments

# Split source--reference parallel file and get alignments

python ${scripts_dir}/force_align.py -c ${tmp_dir}/source_reference_parallel_corpus -fa ${scripts_dir}/fast_align/build \
    -fp ${align_dir}/align.s2t -fe ${align_dir}/align.s2t.err -rp ${align_dir}/align.t2s -re ${align_dir}/align.t2s.err \
    -he grow-diag-final-and -o ${tmp_dir}/source_reference_alignments

#
# Get translation--reference token alignments using TER implemented in tercom
#

echo "Computing translation--reference token alignments using tercom."

# First, format corpora for the tercom tool

python ${scripts_dir}/format_tercom.py ${tmp_dir}/preprocess.translation.tok ${tmp_dir}/preprocess.translation.tercom
python ${scripts_dir}/format_tercom.py ${tmp_dir}/preprocess.reference.tok ${tmp_dir}/preprocess.reference.tercom

# Second, run tercom with shifts deactivated (that's how they did/do at WMT QE tasks)

LC_ALL=${tl_locale} java -jar ${scripts_dir}/tercom-0.7.25/tercom.7.25.jar \
    -r ${tmp_dir}/preprocess.reference.tercom -h ${tmp_dir}/preprocess.translation.tercom \
    -n ${tmp_dir}/reference_translation_tercom -d 0 > ${tmp_dir}/tercom.log 2> /dev/null

# Finally, edit the alignments

python ${scripts_dir}/edit_alignments.py ${tmp_dir}/reference_translation_tercom.xml \
    ${tmp_dir}/preprocess.translation.tok ${tmp_dir}/preprocess.reference.tok ${tmp_dir}/reference_translation_alignments

#
# Generate token-level OK and BAD tags based on alignments
#

echo "Generating token-level OK/BAD tags."

python ${scripts_dir}/generate_tags.py \
    --in-source-tokens ${tmp_dir}/preprocess.source.tok \
    --in-mt-tokens ${tmp_dir}/preprocess.translation.tok \
    --in-pe-tokens ${tmp_dir}/preprocess.reference.tok \
    --in-source-pe-alignments ${tmp_dir}/source_reference_alignments \
    --in-pe-mt-alignments ${tmp_dir}/reference_translation_alignments \
    --out-source-tags ${tmp_dir}/source.tags \
    --out-target-tags ${tmp_dir}/translation.tags \
    --fluency-rule normal
mv ${tmp_dir}/source.tags ${output_directory}/source.tags
mv ${tmp_dir}/translation.tags ${output_directory}/translation.tags

#
# Generate TER sentence-level scores using tercom, allowing shifts
#

echo "Generating sentence-level TER scores."

# Run tercom

LC_ALL=${tl_locale} java -jar ${scripts_dir}/tercom-0.7.25/tercom.7.25.jar \
    -r ${tmp_dir}/preprocess.reference.tercom -h ${tmp_dir}/preprocess.translation.tercom \
    -n ${tmp_dir}/reference_translation_tercom > ${tmp_dir}/tercom.log 2> /dev/null

# Extract TER scores

tail -n+3 ${tmp_dir}/reference_translation_tercom.ter | cut -d" " -f4 > ${output_directory}/translation.ter

#
# Final sanity check and build JSON file
#

echo "Generating single JSON file."

python ${scripts_dir}/build_json.py \
    -s ${source_corpus} \
    -t ${translation_corpus} \
    -stok ${tmp_dir}/preprocess.source.tok \
    -ttok ${tmp_dir}/preprocess.translation.tok \
    -stag ${output_directory}/source.tags \
    -ttag ${output_directory}/translation.tags \
    -bleu ${output_directory}/translation.bleu \
    -ter ${output_directory}/translation.ter \
    -chrf ${output_directory}/translation.chrf \
    -ml 500 \
    -o ${output_directory}/mtqe_corpus.json

#
# Cleanup
#

echo "Removing intermediate files."
rm -rf ${tmp_dir}
