import re, sys
xml = open("prof_unpacked/word/document.xml", encoding="utf-8").read()
# split into paragraphs
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
out=[]
for p in paras:
    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)
    delt = re.findall(r"<w:delText[^>]*>(.*?)</w:delText>", p, re.S)
    line = "".join(texts)
    line = (line.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
                .replace("&#x2019;","’").replace("&#x201C;","“").replace("&#x201D;","”")
                .replace("&#x2018;","‘").replace("&#x2013;","-").replace("&#x2014;","--"))
    if delt:
        d="".join(delt)
        line += "  [DEL: "+d[:120]+"]"
    if line.strip():
        out.append(line)
open("prof_extract.txt","w",encoding="utf-8").write("\n".join(out))
print(len(out),"paragraphs")
