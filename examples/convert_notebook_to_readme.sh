#/bin/bash
jupyter-nbconvert --to markdown MeshoidTest.ipynb
cp ../README_base.md ../README.md
cat MeshoidTest.md >> ../README.md
mv MeshoidTest_files/* ../MeshoidTest_files


jupyter-nbconvert --to rst MeshoidTest.ipynb
cp MeshoidTest.rst ../docs/source/Walkthrough.rst
cp -r ../MeshoidTest_files/* ../docs/source/MeshoidTest_files
