#/bin/bash
jupyter-nbconvert --to markdown MeshoidTest.ipynb
cp ../README_base.md ../README.md
cat MeshoidTest.md >> ../README.md
mv MeshoidTest_files/* ../MeshoidTest_files
