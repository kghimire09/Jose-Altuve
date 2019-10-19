import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
fig, ax = plt.subplots()
jose_altuve['type']=jose_altuve['type'].map({'S':1,'B':0})
jose_altuve=jose_altuve.dropna(subset=  ['type','plate_x','plate_z'])
plt.scatter(x=jose_altuve.plate_x,y=jose_altuve.plate_z,     c=jose_altuve.type, cmap=plt.cm.coolwarm, alpha=0.5)
training_set, validation_set=         train_test_split(jose_altuve, random_state=1)
classifier=SVC(kernel='rbf', gamma=100, C=100)
classifier.fit(training_set[['plate_x','plate_z']],   training_set['type'])
draw_boundary(ax, classifier)
accuracy= classifier.score(training_set[['plate_x','plate_z']], training_set['type'])
print(accuracy)
ax.set_ylim(-2,6)
ax.set_xlim(-3,3)
plt.show()


