import zipfile, re

z = zipfile.ZipFile('D:/Olivia/Artigo_SBBD_Regina_JM.docx')
with z.open('word/document.xml') as f:
    content = f.read().decode('utf-8')

para_pattern = re.compile(r'<w:p[ >].*?</w:p>', re.DOTALL)
paras = para_pattern.findall(content)
for i, p in enumerate(paras):
    if 'commentReference' in p or 'commentRangeStart' in p:
        texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.DOTALL)
        comment_ids = re.findall(r'w:id="([0-9]+)"', p)
        print(f'--- Para {i} (comment ids: {comment_ids}) ---')
        print(' '.join(texts[:15]))
        print()
