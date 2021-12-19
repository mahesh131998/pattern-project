### Pattern Recognition Project

```
Team number : 45
Topic : Baseline ML and DL models and improvisation
Members : Mahesh Desai(50419649), Sagar Sonawane(50431730), Shriram Ravi (50419944)
```

## **HOW TO CHANGE THE PARAMETERS TO RUN THE CODE**
# **LDAM**
- `!pip install tensorboardX`
- `cd LDAM-DRW`
- `python mnist_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --epochs 10 --workers 2 --introduce_noise 1 --noise_ratio 20 --asymmetric_noise 1 --imbalance_data 1`
- To put noise `introduce_noise 1` – to introduce noise or `introduce_noise 0` for no noise
- To put assymetric  noise `asymmetric_noise 1` or `asymmetric_noise 0` for symmetric noise
- To put imbalance dataset `imbalance_data 1` or `imbalance_data 0` for balanced dataset
- The output is displayed in below cells for all the cases.
# **Symmetric cross-entropy Learning (SL)**
- *This is in .ipynb file*
- To run the file just press ctrl+F9 or in menu bar under Runtime select Run all.
- All the parameters are present in the second cell.
- To add the noise Put `noise_ratio = 20` or some random number else `noise_ratio = 0` for no noise.
- To put assymetric noise put `asym = True` or `asym = False` for symmetric noise
- To imbalance the dataset put `dasm = True` or `dasm = False` for balanced dataset
- The output is displayed in last cell.
- While running the code again to check the new cases make sure that all the data, log and model files are deleted for previous case , otherwise it will give out error.
# **Logistic Regression**
- *This is in .ipynb file*
- To run the file just press ctrl+F9 or in menu bar under Runtime select Run all.
- The output is displayed in below cells for all the cases.
# **SVM Model**
- To add the noise put `add_noise = True` else `add_noise = False` for no noise.
- To put assymetric noise put `asym_noise = True` or `asym_noise = False` for symmetric noise
- To imbalance the dataset put `imbalance = True` or `imbalance = False` for balanced dataset
- The output is displayed in below cells for all the cases.
# **VGG19 Model**
- To add the noise put `add_noise = True` else `add_noise = False` for no noise.
- To put assymetric noise put `asym_noise = True` or `asym_noise = False` for symmetric noise
- To imbalance the dataset put `imbalance = True` or `imbalance = False` for balanced dataset
- The output is displayed in below cells for all the cases.
