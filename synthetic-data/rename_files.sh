j=0
for i in ./512x384/*.jpg; do
  #echo $i
  #file_prefix=$(echo ${i} | cut -d'.' -f 2)
  #echo $file_prefix 
  #echo $i
  file_prefix=$(echo ${i%.jpg})
  file_prefix=$(echo ${file_prefix} | cut -d'/' -f 3)
  echo $j
  echo $file_prefix

  mv $i ./512x384_validation/${file_prefix}.jpg
  mv ./512x384_noisy/${file_prefix}.jpg ./512x384_noisy_validation/${file_prefix}.jpg
  ((j++))
  if [[ $j -ge 1000 ]]; then
    break
  fi
 
done


