<h1>python�Ω�����ǲߪ��`�μҲ�</h1>
============================
<h3>numpy</h3>
----------------------------

<p>numpy�䴩���װ}�C���B�⦡�A���L�����ǲ߭n�B�z����ƫ��A�����|�s���x�}���榡�C
<br>
�w�ˤ�k
<pre><code>$ pip install numpy</code></pre>
<br>
<p>�ϥΤ�k
<pre><code>
>>> import numpy as np
>>> x = np.array([1, 2, 3])
>>> x
array([1, 2, 3])
</code></pre>



<h3>sklearn.decomposition.PCA</h3>
----------------------------

<p>sklearn�Opython�����ǲߨ禡�w
<p>PCA�O�����ǲߤ����@�Ӥ�k �٧@�����A�U�W��q�N�O���C���סC
<p>�N�����ת���T���C��C���� �i�H������L�����ǲߺt��k���e���e�B�z
<br>
�w�ˤ�k
<pre><code>$ pip install sklearn</code></pre>
<br>
<p>�ϥΤ�k
<pre><code>
>>> from sklearn.decomposition import PCA
>>> pca=PCA(n_components=i)#PCA 
>>> pca.fit(immatrix)
>>> result=pca.transform(testimage)
</code></pre>