READ ME
HOW TO CHANGE THE PARAMETERS TO RUN THE CODE

LDAM:-
How to run the code-
python mnist_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --epochs 10 --workers 2 --introduce_noise 1 --noise_ratio 20 --asymmetric_noise 1 --imbalance_data 1
To put noise  introduce_noise must be 1 – to introduce noise or 0 for no noise
To put assymetric  noise  asymmetric_noise must be 1 or 0 for symmetric noise
To put imbalance dataset imbalance_data must be 1 or 0 for balanced dataset
The output is displayed in below cells for all the cases.

SL :-
How to run the code-
This is in .ipynb file
To run the file just press ctrl+F9 or in menu bar under Runtime select Run all.
All the parameters are present in the second cell.
To add the noise  Put noise_ratio= 20 or some random number else 0  for no noise.
To put assymetric  noise  Put asym parameter True or False for symmetric noise
To imbalance the dataset  Put dasm parameter True or False for balanced dataset
The output is displayed in last cell.
While running the code again to check the new cases make sure that all the data, log and model files are deleted for previous case , otherwise it will give out error.

Logistic Regression:-
How to run the code-
This is in .ipynb file
To run the file just press ctrl+F9 or in menu bar under Runtime select Run all.
The output is displayed in below cells for all the cases.


SVM Model:-
How to run the code-
To add the noise  Put add_noise= True else False for no noise.
To put assymetric  noise  Put asym_noice parameter True or False for symmetric noise
To imbalance the dataset  Put dasm parameter True or False for balanced dataset
The output is displayed in below cells for all the cases.



VGG19 Model:-
How to run the code-
