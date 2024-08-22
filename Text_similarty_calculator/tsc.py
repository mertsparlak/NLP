import spacy
# türkçe dil modeli kullanmak için terminalden indirmemiz gerekiyor kopyalayıp yapıştırmanız için bırakıyorum:
# pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl

nlp=spacy.load("tr_core_news_lg")
# modeli ingilizce yapmak istiyorsanız şunu kullanabilirsiniz :"en_core_web_lg"

w1="sarı"
w2="lacivert"

w1=nlp.vocab[w1]
w2=nlp.vocab[w2]

print(w1.similarity(w2))
