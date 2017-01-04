<h1>python用於機器學習的常用模組</h1>
============================
<h3>numpy</h3>
----------------------------

<p>numpy支援維度陣列的運算式，不過機器學習要處理的資料型態往往會存成矩陣的格式。
<br>
安裝方法
<pre><code>$ pip install numpy</code></pre>
<br>
<p>使用方法
<pre><code>
>>> import numpy as np
>>> x = np.array([1, 2, 3])
>>> x
array([1, 2, 3])
</code></pre>



<h3>sklearn.decomposition.PCA</h3>
----------------------------

<p>sklearn是python機器學習函式庫
<p>PCA是機器學習中的一個方法 稱作降維，顧名思義就是降低維度。
<p>將高維度的資訊降低到低維度 可以做為其他機器學習演算法之前的前處理
<br>
安裝方法
<pre><code>$ pip install sklearn</code></pre>
<br>
<p>使用方法
<pre><code>
>>> from sklearn.decomposition import PCA
>>> pca=PCA(n_components=i)#PCA 
>>> pca.fit(immatrix)
>>> result=pca.transform(testimage)
</code></pre>