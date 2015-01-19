
#importing library and replacing the current column names with names
import pandas as pd
df = pd.read_csv('/Users/kuttush/Desktop/Spongebob/Thinkful/DataScience/Unit4/NaiveBayes/ideal_weight.csv', names=['id', 'sex', 'actual', 'ideal', 'diff'], header=0)


#removing the single quote from the sex column entries
df['sex'] = df['sex'].map(lambda x: x.replace("'",""))
#df['sex'] = df['sex'].map(lambda x: x.rstrip('').lstrip('')
'''A = df['sex'].tolist()
B=[]
for i in range(len(A)):
	test = A[i].strip(' ')
	B.append(test)
df['sex'] = B'''

#plot the distribution of actual and ideal weight using stacked histogram
import matplotlib.pyplot as plt
plt.figure()
plt.hist([df['actual'], df['ideal']], histtype='bar', stacked=True)
plt.show()

#plot the distribution of difference in weigh
plt.figure()
plt.hist(df['diff'], histtype='bar')
plt.show()

#map sex to categorical variable and checking
A = pd.Categorical(df['sex'].tolist())
print(len(A[A=='Male']))
print(len(A[A=='Female']))

'''there are 63 males and 119 females.'''


#Fit a Naive Bayes classifier of sex to actual weight, ideal weight, and diff.
data = df[['actual', 'ideal', 'diff']]
target = A
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)
print("Number of mislabeled points out of a total %d points: %d" %(data.shape[0], (target != y_pred).sum()))

'''Number of mislabeled points are 14 out of a total of 182 points. '''
