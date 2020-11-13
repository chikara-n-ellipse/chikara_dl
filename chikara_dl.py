import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
  y = sigmoid(x)
  return y*(1-y)

def relu(x):
  return np.maximum(x,0)

def relu_deriv(x):
  return np.where(x>=0,1,0)

def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c)
  Z = np.sum(exp_x,axis=0)
  y = exp_x / Z
  return y

def softmax_deriv(x): #return a matrix
  y = softmax(x) #(K,N)
  def get_y_yy(y):
    return np.diag(y)-y.dot(y.T)
  Y = np.apply_along_axis(get_y_yy, 0, y)
  return Y

def crossentropy(y,t):
  return -np.sum(t*np.log(y),axis=0)

def chikara():
  print("Hello, Chikara!")

def get_flattened_dyad(x,y=None):# x : (N, I)
  def f(a,b=None):# a,b : vector
    A = a[:,np.newaxis]
    dyad = A.dot(A.T) if y==None else A.dot(b[:,np.newaxis].T)
    fd = dyad[np.triu(np.ones(dyad.shape))==1]
    return fd
  X = np.apply_along_axis(f,1,x)
  return X

def get_square_extension_data(x):
  fd = get_flattened_dyad(x)
  X = np.concatenate([x, fd], 1)
  return X

def getXOR():
  a = np.random.normal(0,1,size=(1000,2))
  b = np.where(a[:,0]*a[:,1]<0,1,0)[:,np.newaxis]
  data_labels = np.concatenate([a,b],axis=1)
  return  data_labels

class Dense:

  def __init__(self,units,activation='relu'):
    self.units = units
    self.activation = activation
    self.epsilon = 0.001
    self.alpha = 0.001
    self.myu = 0.001
    self.preDifWeight = 0
    self.total_input = None
    self.output = None
    self.delta = None
    self.delWeight = None
    self.difWeight = None
  
  def activate(self,u):
    if self.activation=='relu':
      return relu(u)
    elif self.activation=='sigmoid':
      return sigmoid(u)
  
  def activate_deriv(self,u):
    if self.activation=='relu':
      return relu_deriv(u)
    elif self.activation=='sigmoid':
      return sigmoid_deriv(u)

  def initializeWeight(self,previous_units):
    variance = np.sqrt(2/previous_units) if self.activation=='relu' else 1/np.sqrt(previous_units) 
    bias = np.zeros((self.units, 1))
    w = np.random.normal(0,variance,size=(self.units, previous_units))
    self.weight = np.concatenate([bias,w],axis=1)#(J,I+1)
  
  def setInOut(self, z): #z(I,N)
    N = z.shape[1]
    ones = np.ones((1,N))
    z = np.concatenate([ones,z],axis=0)#z(I+1,N)
    self.total_input = self.weight.dot(z)#(J,N)
    self.output = self.activate(self.total_input)#(J,N)
  
  def setDelta(self,delta_input):#d,w : next layer delta and weight
    delta, w = delta_input
    self.delta = self.activate_deriv(self.total_input)*((w[:,1:].T).dot(delta))#(J,N)

  def setDelWeight(self,z):
    N = z.shape[1]
    ones = np.ones((1,N))
    z = np.concatenate([ones,z],axis=0)
    self.delWeight = self.delta.dot(z.T)/self.delta.shape[1]
  
  def updateWeight(self):
    self.difWeight = (self.myu)*self.preDifWeight - (self.epsilon)*(self.delWeight + self.alpha*self.weight)
    self.weight += self.difWeight
    self.preDifWeight = self.difWeight

class Softmax:

  def __init__(self,units,activation='softmax'):
    self.units = units
    self.activation = activation
    self.preDifWeight = 0
    self.epsilon=0.001
    self.alpha=0.001
    self.myu=0.001
    self.total_input = None
    self.output = None
    self.delta = None
    self.delWeight = None
    self.difWeight = None
  
  def activate(self,u):
    if self.activation=='softmax':
      return softmax(u)
  
  def activate_deriv(self,u):#return a matrix 
    if self.activation=='softmax':
      return softmax_deriv(u)

  def initializeWeight(self,previous_units):
    variance = np.sqrt(2/previous_units) if self.activation=='relu' else 1/np.sqrt(previous_units) 
    bias = np.zeros((self.units - 1, 1))
    w = np.random.normal(0, variance, size=(self.units - 1, previous_units))
    self.weight = np.concatenate([bias,w],axis=1)#(J-1,I+1)
  
  def setInOut(self,z):#z(I,N)
    N = z.shape[1]
    ones = np.ones((1,N))
    z = np.concatenate([ones,z],axis=0)#z(I+1,N)
    self.total_input = self.weight.dot(z)  #(J-1,N)
    self.output = self.activate(np.concatenate([np.zeros((1,N)),self.total_input],axis=0)) #(J,N)
  
  def setDelta(self,delta_input): # d:supervisor(labels)
    d = delta_input
    y = self.output
    self.delta = (y - d)[1:]  #(J-1,N), set J-1 Dim to  be multiplied by w when back_propagation

  def setDelWeight(self,z):
    N = z.shape[1]
    ones = np.ones((1,N))
    z = np.concatenate([ones,z],axis=0)
    self.delWeight = self.delta.dot(z.T)/self.delta.shape[1]
  
  def updateWeight(self):
    self.difWeight = (self.myu)*self.preDifWeight - (self.epsilon)*(self.delWeight + self.alpha*self.weight)
    self.weight += self.difWeight
    self.preDifWeight = self.difWeight
  

class Sequential:
  def __init__(self, layers, input_size, loss, batch_size=50, metrics=['accuracy'], epsilon=0.001,alpha=0.001,myu=0.001,is_debug=False):
    self.layers = layers
    self.layer_num = len(self.layers)
    self.input_size = input_size
    self.loss = loss
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.alpha = alpha
    self.myu = myu
    self.is_debug = is_debug
    self.loss_value = None
    self.history = None

    u = self.input_size
    for layer in self.layers:
      layer.initializeWeight(u)
      u = layer.units
    for layer in self.layers:
      layer.epsilon = self.epsilon
      layer.alpha = self.alpha
      layer.myu = self.myu
    self.output_size = u
  
  def predict(self,data):
    z = data.T
    for layer in self.layers:
      layer.setInOut(z)
      z = layer.output
    self.output = z
    return self.output

  
  def getMiniBatchData(self,data,labels):
    mbdl_index = np.random.permutation(data.shape[0])[:self.batch_size]
    mb_data = data[mbdl_index]
    mb_labels = labels[mbdl_index]
    return (mb_data, mb_labels)
  
  def fit(self, data, labels, epochs=100,validation_data = None):
    
    N = data.shape[0]
    batch_size = self.batch_size
    self.printAllState("start") if self.is_debug == True else 0
    for epoch in range(epochs):
      loss, accuracy, v_loss, v_accuracy = self.setHistory(epoch, data, labels, validation_data)
      if epoch%100==0:
        if validation_data!=None:
          print("MCE:{:3.5f} ACC:{:1.3f} v_MCE:{:3.5f} v_ACC:{:1.3f}".format(loss, accuracy, v_loss, v_accuracy))
          
        else:
          print("MCE:{:3.5f} ACC:{:1.3f}".format(loss, accuracy))
      for i in range(int(N/batch_size)):
        mb_data, mb_labels = self.getMiniBatchData(data,labels)
        print("mb_data",mb_data) if self.is_debug == True else 0
        print("mb_labels",mb_labels) if self.is_debug == True else 0
        self.predict(mb_data)
        self.setLossValue(self.output, mb_labels)
        self.setAccuracy(mb_labels)
        if self.is_debug == True:
          self.printAllState("after predict")
          print("MCE:{:3.5f} ACC:{:1.3f}".format(self.loss_value.mean(),self.accuracy))
        self.update(mb_data,mb_labels)
        self.printAllState("after update") if self.is_debug == True else 0
    self.showGraph()
  def setHistory(self, epoch, data, labels, validation_data):
    y = self.predict(data)
    self.setLossValue(y,labels)
    loss = self.loss_value.mean()
    self.setAccuracy(labels)
    accuracy = self.accuracy
    if validation_data!=None:
      (test_data, test_labels) = validation_data
      y = self.predict(test_data)
      self.setLossValue(y,test_labels)
      v_loss = self.loss_value.mean()
      self.setAccuracy(test_labels)
      v_accuracy = self.accuracy
    else:
      v_loss, v_accuracy = (None,None)
    if self.history is None:
      self.history = np.array([[epoch, loss, accuracy, v_loss, v_accuracy]])
    else:
      a = np.array([[epoch, loss, accuracy, v_loss, v_accuracy]])
      self.history = np.append(self.history,a,axis=0)
    return loss, accuracy, v_loss, v_accuracy

  def getValidationLossAndAccuracy(validation_data):
    (test_data, test_labels) = validation_data
    if self.loss == 'crossentropy':
      y = self.predict(test_data)
      validation_loss = crossentropy(y,test_labels.T)
    else:
      return None
    return (validation_loss, validation_accuracy)
  
  def setLossValue(self, y, labels):
    d = labels.T
    if self.loss == 'crossentropy':
      self.loss_value = crossentropy(y,d)
  
  def setAccuracy(self,labels):
    b = np.argmax(self.output,axis=0)==np.argmax(labels.T,axis=0) #(K,N)->(N)
    self.accuracy = np.count_nonzero(b)/b.shape[0] # num_of_true / N

  def update(self, mb_data, mb_labels):
    d = mb_labels.T #d:superviser
    batch_size = self.output.shape[1]
    #back_propagation
    delta_input = d
    for layer in reversed(self.layers):
      layer.setDelta(delta_input)
      delta_input = (layer.delta, layer.weight)
    self.printAllState("after delta back propagation") if self.is_debug == True else 0
    z = mb_data.T
    for layer in self.layers:
      layer.setDelWeight(z)
      layer.updateWeight()
      z = layer.output
    self.printAllState("after update weight") if self.is_debug == True else 0

  def printAllState(self,str):
    print("---------------------------------")
    print(str)
    for l,layer in enumerate(self.layers):
      print("layer:",l)
      print("w")
      print(layer.weight)
      print("u")
      print(layer.total_input)
      print("z")
      print(layer.output)
      print("delta")
      print(layer.delta)
      print("delWeight")
      print(layer.delWeight)
      print("difWeight")
      print(layer.difWeight)
    input()
  
  def showGraph(self):
    fig, ax = plt.subplots(2, 1)
    line_loss, = ax[0].plot(self.history[:,0],self.history[:,1], color='blue')
    line_loss, = ax[1].plot(self.history[:,0],self.history[:,2], color='blue',linestyle='dashed')
    line_loss, = ax[0].plot(self.history[:,0],self.history[:,3], color='red')
    line_loss, = ax[1].plot(self.history[:,0],self.history[:,4], color='red',linestyle='dashed')
    
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].grid()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].grid()
    plt.show()

