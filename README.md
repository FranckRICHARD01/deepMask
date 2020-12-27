# deepMask

**USAGE:**
```
docker run -it -v /tmp:/tmp deepmask /app/inference.py \
                                            $PATIENT_ID \
                                            /tmp/T1.nii.gz /tmp/FLAIR.nii.gz \
                                            /tmp
```