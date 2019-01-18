# OSDA-EM
Open Set Domain Adaptation with Entropy Minimization

# Prerequist 
These codes work on Pytorch 0.4.0 and Python3.6

# Download
Datasets include BCIS, please downloaded them from webside and unzipped them into yourpath/data/

# Run

Run the open set domain adaptation from BING to CALTECH256 by:
1) set the source_name and target_name in osda_em.py as
   source_name = "bing"        
   target_name = "caltech256"  
2) run osda_em.py
   python3 mnist2svhn_unsup.py
