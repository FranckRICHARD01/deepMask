#!/bin/bash

# usage: $0 <cases>

source ${NOELSOFT_DIR}/BashTools/noel_do_cmd
source ${NOELSOFT_DIR}/BashTools/noel_do_echo
source ${NOELSOFT_DIR}/BashTools/noel_do_cmd_new

CRFDIR=/host/silius/local_raid/ravnoor/01_Projects/12_deepMask/src/densecrf/dense3dCrf
export PATH=${CRFDIR}:${PATH}
# for id in */*_maskpred.nii.gz; do echo $id; out=$(echo $id | sed s'/.nii.gz/.mnc/g'); echo $out; done

OUTDIR=/host/silius/local_raid/ravnoor/01_Projects/12_deepMask/src/predictions/vnet.masker.20180316_0441
# OUTDIR=/host/silius/local_raid/ravnoor/01_Projects/12_deepMask/src_T1/predictions/vnet.masker_T1.20180316_2128
INDIR=/host/hamlet/local_raid/data/ravnoorX/deepMask/data/EXT_3T
# INDIR=/host/silius/local_raid/ravnoor/01_Projects/12_deepMask/data/BEP
CROSSDIR=/data/noel/noel8/CrossSite_FCD/02_T1_Processing/deepMask
ID=$1 # FLP_006

X=`fslval ${INDIR}/${ID}/T1.nii.gz dim1`
Y=`fslval ${INDIR}/${ID}/T1.nii.gz dim2`
Z=`fslval ${INDIR}/${ID}/T1.nii.gz dim3`

if [ ! -d ${CROSSDIR}/${ID} ]; then
  mkdir ${CROSSDIR}/${ID}
fi

cp -f ${CRFDIR}/config.txt /tmp/${ID}_config.txt
# cp -f ${CRFDIR}/config_T1_only.txt /tmp/${ID}_config.txt

sed -i -e "s|OUTDIR_PLACEHOLDER|${OUTDIR}|g" /tmp/${ID}_config.txt
sed -i -e "s|INDIR_PLACEHOLDER|${INDIR}|g" /tmp/${ID}_config.txt
sed -i -e "s|ID_PLACEHOLDER|${ID}|g" /tmp/${ID}_config.txt

sed -i -e "s|X_PLACEHOLDER|${X}|g" /tmp/${ID}_config.txt
sed -i -e "s|Y_PLACEHOLDER|${Y}|g" /tmp/${ID}_config.txt
sed -i -e "s|Z_PLACEHOLDER|${Z}|g" /tmp/${ID}_config.txt

noel_do_cmd_new ${CROSSDIR}/${ID}/${ID}_vnet_maskpred.nii.gz cp ${OUTDIR}/${ID}/${ID}_vnet_maskpred.nii.gz ${CROSSDIR}/${ID}/
noel_do_cmd_new ${CROSSDIR}/${ID}/${ID}_vnet_maskpred.mnc.gz AimsFileConvert -i ${CROSSDIR}/${ID}/${ID}_vnet_maskpred.nii.gz -o ${CROSSDIR}/${ID}/${ID}_vnet_maskpred.mnc
gzip -f ${CROSSDIR}/${ID}/${ID}_vnet_maskpred.mnc

noel_do_cmd_new ${OUTDIR}/${ID}/${ID}_denseCrf3dSegmMap.nii.gz dense3DCrfInferenceOnNiis -c /tmp/${ID}_config.txt

noel_do_cmd_new ${CROSSDIR}/${ID}/${ID}_denseCrf3dSegmMap.mnc.gz AimsFileConvert -i ${OUTDIR}/${ID}/${ID}_denseCrf3dSegmMap.nii.gz -o ${CROSSDIR}/${ID}/${ID}_denseCrf3dSegmMap.mnc

gzip -f ${CROSSDIR}/${ID}/${ID}_denseCrf3dSegmMap.mnc

rm -f ${CROSSDIR}/${ID}/*.minf

# rm -f ${INDIR}/${ID}/${TO}.nii.gz
