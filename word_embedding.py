from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')
f=open('anger','r')
x=f.readlines()
count=1
for a in x:
  senfile=open(str(count)+'.npy','wb')
  s=a.split(' ')
  for w in s:
    sentence_embedding = model.encode(w)
    np.save(senfile,sentence_embedding)
  senfile.close()
  count+=1

