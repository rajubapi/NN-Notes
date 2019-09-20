for file in **/*.ipynb
do
  printf $file
  printf '\n'
  jupyter nbconvert --to markdown $file
done
