import sys, re
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\Quang Huy\Downloads\Báo cáo AI_SamSung.rtf'
with open(path, 'rb') as f:
    raw = f.read()

# In RTF, fields look like:
# {\field{\*\fldinst { HYPERLINK \l "_Toc244334270" }}{\fldrslt {LỜI CẢM ƠN\tab 1}}}
# or PAGEREF

def decode_rtf_fragment(b):
    text = b.decode('latin1', errors='ignore')
    def u_rep(m):
        c = int(m.group(1))
        return chr(c if c >= 0 else c + 65536)
    text = re.sub(r'\\u([0-9\-]+)\??', u_rep, text)
    def h_rep(m):
        try:
            return bytes.fromhex(m.group(1)).decode('windows-1258')
        except:
            return ''
    text = re.sub(r'\\\'([0-9a-fA-F]{2})', h_rep, text)
    text = re.sub(r'\\[a-zA-Z0-9\-]+ ?', ' ', text)
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Find all fldinst HYPERLINK and fldrslt
matches = re.findall(rb'HYPERLINK \\l \"(_Toc\d+)\"[^\}]*\}\}\{\\fldrslt\s*\{?([^\}]+)', raw)
print(f'Matches count: {len(matches)}')
for tag, res in matches:
    clean = decode_rtf_fragment(res)
    if clean:
        print(f'{tag.decode()}: {clean}')

# If matches is small, let's also search for paragraphs with _Toc
if len(matches) < 5:
    print('\n--- Searching for _Toc in full raw text ---')
    tocs = re.findall(rb'(_Toc\d+)[^\\\{]{0,50}([^\r\n]{3,80})', raw)
    for t, content in tocs[:60]:
        print(f'{t.decode()}: {decode_rtf_fragment(content)}')
