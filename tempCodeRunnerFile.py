X = np.array(df.drop(columns=['label']))
y = np.array([df["label"]]).T

# Select the set with the best n attributes using the method pca
n = 0
score = 0
for i in range(1,21):
	print("Lan", i)
	pca = decomposition.PCA(n_components=i)
	pca.fit(X)
	# print(pca)
	Xbar = pca.transform(X)  # ap dung giam kich thuoc cho X.
	X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=False)
	model = DecisionTreeClassifier(criterion = "gini")
	model = model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	d = 0
	trained_score = accuracy_score(y_test, y_pred)
	print("Score =", trained_score)
	if(trained_score >= score):
		score = trained_score
		n = i

# Use the selected set of n good attributes to create a new train and test set
print("N_components:", n)
main_pca = decomposition.PCA(n_components=n)
main_pca.fit(X)
Xbar1 = main_pca.transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xbar1, y, test_size=0.3, shuffle=True)

# Use CART to train model
mainModel = DecisionTreeClassifier(criterion = "gini")
mainModel.fit(X_train1, y_train1)
y_pred1 = mainModel.predict(X_test1)


cnf_matrix = confusion_matrix(y_test1, y_pred1)
print('Confusion matrix:')
print(cnf_matrix)

# confusion matrix to precision + recall
def cm2pr_binary(cm):
    p = cm[0,0]/np.sum(cm[:,0])
    r = cm[0,0]/np.sum(cm[0])
    return (p, r)

# model evaluation
acc = accuracy_score(y_test1, y_pred1) #do cxac
# precision = (TP)/(TP+FP)
# recall = (TP)/(TP+FN)
precision,recall = cm2pr_binary(cnf_matrix)
# f1-score is a combination of precision and recall
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy = {0:.2f}'.format(acc))
print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1-score = {0:.2f}'.format(f1_score))

#Input data from the user side and then predict male or female
Predict_from_user_data(main_pca, mainModel)
