# CAPER

**Paper**: Jing Zhu, Danai Koutra, Mark Heimann
CAPER: Coarsen, Align, Project, Refine
A General Multilevel Framework for Network Alignment

*Link*: TBD  

<p align="center">
<img src="https://raw.githubusercontent.com/GemsLab/CAPER/master/approach.png" width="700"  alt="CAPER overview">
</p>

`python3 nhem.py --data ../data/arenas/arenas800-3/arenas_combined_edges.txt --coarsen-level 3 --output-path test.pkl` 

`python main.py --true_align data/arenas/arenas800-3/arenas_edges-mapping-permutation.txt --combined_graph coarsening/test.pkl --embmethod xnetMF --alignmethod REGAL --refinemethod RefiNA  --coarsen`