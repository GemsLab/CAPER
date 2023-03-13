# CAPER

**Paper**: Jing Zhu, Danai Koutra, Mark Heimann
CAPER: Coarsen, Align, Project, Refine
A General Multilevel Framework for Network Alignment

*Link*:  https://arxiv.org/abs/2208.10682

<p align="center">
<img src="https://raw.githubusercontent.com/GemsLab/CAPER/master/approach.png" width="700"  alt="CAPER overview">
</p>


**Citation (bibtex)**:
```
@inproceedings{caper,
  title={CAPER: Coarsen, Align, Project, Refine - A General Multilevel Framework for Network Alignment},
  author={Zhu, Jing and Koutra, Danai and Heimann Mark},
  booktitle={CIKM},
  year={2022}
}
```
## Usage
For graph coarsening: 

```python3 nhem.py --data ../data/arenas/arenas800-3/arenas_combined_edges.txt --coarsen-level 3 --output-path test.pkl```

For running alignment on coarsened graphs: 

```python main.py --true_align data/arenas/arenas800-3/arenas_edges-mapping-permutation.txt --combined_graph coarsening/test.pkl --embmethod xnetMF --alignmethod REGAL --refinemethod RefiNA  --coarsen```

# Question & troubleshooting

If you encounter any problems running the code, pls feel free to contact Jing Zhu (jingzhuu@umich.edu)
