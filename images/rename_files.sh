for i in ./ground_truths/*.JPG; do
  #echo $i
  #file_prefix=$(echo ${i} | cut -d'.' -f 2)
  #echo $file_prefix 
  echo $i
  file_prefix=$(echo ${i%_mean.JPG})
  echo $file_prefix
  mv $i ${file_prefix}.JPG 


done

for i in ./noisy/*.JPG; do
  #echo $i
  #file_prefix=$(echo ${i} | cut -d'.' -f 2)
  #echo $file_prefix 
  echo $i
  file_prefix=$(echo ${i%_real.JPG})
  echo $file_prefix
  mv $i ${file_prefix}.JPG  


done

