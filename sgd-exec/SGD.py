import matplotlib.pyplot as plt
import numpy as np


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# ########################################

class Base:

    def update(self,w , point,label):
        raise NotImplementedError()

    def fit(self, X, y, sample_weight=None):

        temp = np.ones([X.shape[0],1])
        X=np.hstack((X,temp))
        self.w = np.zeros(X.shape[1])

        for i in range(300):
            #self.w =(i+1)/(i+2) * self.w

            #
            s=np.random.choice(X.shape[0])

            #update rule :
            self.w=self.update(self.w, X[s, :] , y[s])

            if i % 50 == 1 :
                xx = np.linspace(-2, 10)
                a = -self.w[0] / self.w[1]
                bias = self.w[2]
                yy = a * xx - (bias) / self.w[1]
                lines = al.plot(xx, yy, 'k-')
                al.set_title(self.name)
                fig.ginput()
                lines.pop(0).remove()

        return self




    def predict(self, X):
        X = np.hstack((X, np.ones([X.shape[0],1])))
        return np.sign(X.dot(self.w.T))


# TODO : write the update rule for each algorithm !!
# maybe need to chane learning step or use learning step decay .
class SVM (Base):
    name = "SVM"

    def update(self,w,point,label):
      ???????
        return w




class Perceptron (Base):
    name="Perceptron"

    def update(self,w,point,label):
       ??????
        return w

class Logistic (Base):

    name="Logistic"

    def update(self,w,point,label):
      ??????
        return w



# #
N=100
s = np.random.normal(0, 1,(N,2))
X1=s+[2,2]
X2=s+[5,5]

X=np.concatenate([X1,X2])
y=np.concatenate([-1*np.ones(N),1*np.ones(N)])

#figure = plt.figure()
fig , al = plt.subplots(1)
al.plot(X1[:, 0], X1[:, 1], '*r', X2[:, 0], X2[:, 1],'^b')  # ,X3[:,0],X3[:,1],'+y',X4[:,0],X4[:,1],'og',X5[:,0],X5[:,1],'''''')


clf1=SVM()
clf2=Perceptron()
clf3=Logistic()

model = (clf1,clf2,clf3)


models = (clf.fit(X, y) for clf in model)

# title for the plots
titles = (  'SVM',
            'Perceptron',
            'Loggistic Regression')

# Set-up 2x2 grid for plotting.
fig2, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

xx, yy = make_meshgrid(X[:, 0], X[:, 1])

for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
