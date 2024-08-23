import tkinter as tk
import nltk
from textblob import TextBlob
from newspaper import Article
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model="savasy/bert-base-turkish-sentiment-cased")

def summarize():
    url=utext.get("1.0","end").strip()
    article=Article(url)

    article.download()
    article.parse()

    article.nlp()

    title.config(state="normal")
    author.config(state="normal")
    summary.config(state="normal")
    datepb.config(state="normal")
    sentiment.config(state="normal")
    
    title.delete("1.0","end")
    title.insert("1.0",article.title)
    
    author.delete("1.0","end")
    author.insert("1.0",article.authors)
    
    datepb.delete("1.0","end")
    datepb.insert("1.0",article.publish_date)
    
    summary.delete("1.0","end")
    summary.insert("1.0",article.summary)
    
    sntmnt=classifier(article.text)
    
    sentiment.delete("1.0","end")
    sentiment.insert("1.0", f"Objektiflik: {'pozitif' if sntmnt[0]['label'] == 'LABEL_2' else 'negatif' if sntmnt[0]['label'] == 'LABEL_0' else 'nötr'}")
    
    title.config(state="disabled")
    author.config(state="disabled")
    summary.config(state="disabled")
    datepb.config(state="disabled")
    sentiment.config(state="disabled")
    

    analysis=TextBlob(analysis.text)
    print(analysis.polarity) 

root=tk.Tk()
root.title("Haber Özetleyici")
root.geometry("1000x550")

tlabel=tk.Label(root,text="Başlık")
tlabel.pack()

title=tk.Text(root,height=1,width=110)
title.config(state="disabled",bg="#dddddd")
title.pack()

alabel=tk.Label(root,text="Yazar")
alabel.pack()

author=tk.Text(root,height=1,width=110)
author.config(state="disabled",bg="#dddddd")
author.pack()

plabel=tk.Label(root,text="Yayın tarihi")
plabel.pack()

datepb=tk.Text(root,height=1,width=110)
datepb.config(state="disabled",bg="#dddddd")
datepb.pack()

slabel=tk.Label(root,text="Özet")
slabel.pack()

summary=tk.Text(root,height=15,width=110)
summary.config(state="disabled",bg="#dddddd")
summary.pack()

selabel=tk.Label(root,text="Objektiflik")
selabel.pack()

sentiment=tk.Text(root,height=1,width=110)
sentiment.config(state="disabled",bg="#dddddd")
sentiment.pack()

ulabel=tk.Label(root,text="URL")
ulabel.pack()

utext=tk.Text(root,height=1,width=110)
utext.pack()

btn=tk.Button(root,text="Özetle",command=summarize)
btn.pack()

root.mainloop()

