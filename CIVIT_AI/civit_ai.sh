#!/bin/sh

civit_ai=$1 

CWDPATH=$(pwd)
mkdir $CWDPATH/CIVIT_AI/safetensors
cd $CWDPATH/CIVIT_AI/safetensors

mkdir $civit_ai
cd $civit_ai
wget https://civitai.com/api/download/models/$civit_ai --content-disposition

model_name=$(ls -l | awk '{print $9}')
model_name=${model_name//$'\n'/}
model_name2=${model_name//$'.safetensors'/}

eval "$(conda shell.bash hook)"
conda activate rave
cd ../..
python convert.py \
    --checkpoint_path "$CWDPATH/CIVIT_AI/safetensors/$civit_ai/$model_name" \
    --dump_path "$CWDPATH/CIVIT_AI/diffusers_models/$civit_ai/$model_name2" \
    --from_safetensors

rm -rf $CWDPATH/CIVIT_AI/safetensors/

echo "Download is done! Check the diffusers_models folder. $model_name"