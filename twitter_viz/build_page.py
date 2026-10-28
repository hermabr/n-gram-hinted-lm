"""Page with the chosen figure inlined, written to $ART/visualize-how-it-works.html."""
import os
import re
from pathlib import Path

HERE = Path(__file__).parent
FIGS = [
    ("g3", "Hint line keeps both colours", "Same as g2, but the joined hint line is pink with orange dashes, so it still shows what it carries."),
    ("g2", "Retrieval hints", "The matched next tokens in the training data become one hint, which is added (⊕) to the stream after the embedding and after each transformer layer."),
]

def scoped(key):
    svg = (HERE / f"retrieval_hints_{key}.svg").read_text()
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{key}-{m.group(1)}"', svg)
    return re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{key}-{m.group(1)})", svg)


def fig(key, title, desc):
    return (f'<figure id="{key}"><header><h2><span>{key}</span> {title}</h2><div class="btns">'
            f'<button onclick="png(\'{key}\')">Download PNG</button><button onclick="svgdl(\'{key}\')">SVG</button></div></header>'
            + (f"<p>{desc}</p>" if desc else "") + f'<div class="frame">{scoped(key)}</div></figure>')


html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval hints: how it works</title>
<style>
body{{margin:0;background:#f6f1f4;color:#1c1c1c;font:16px/1.5 'Helvetica Neue',Helvetica,Arial,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:28px;margin:0 0 6px}} .lede{{color:#75666e;margin:0 0 36px;max-width:820px}}
figure{{margin:0 0 48px}} header{{display:flex;align-items:baseline;justify-content:space-between;gap:16px}}
h2{{font-size:19px;margin:0}} h2 span{{color:#fb64b6}}
p{{color:#75666e;margin:4px 0 12px}}
.frame{{border:1px solid #ead9e1;border-radius:12px;overflow:hidden;line-height:0;box-shadow:0 2px 12px #0000000d;margin-top:8px}}
.frame svg{{width:100%;height:auto}}
button{{font:600 13px/1 inherit;font-family:inherit;padding:7px 12px;border-radius:8px;border:1px solid #ead9e1;background:#fff;cursor:pointer;margin-left:6px}}
button:hover{{border-color:#fb64b6}}
</style></head><body><main>
<h1>Retrieval hints: how it works</h1>
<p class="lede">The context goes to the CPU, which finds it in the training data and counts what came next. Those counts are the hint.
The transformer gets both the context and the hint, and the pink and orange blocks show the gain from the hints.
Each image is 1600×900 (16:9); "Download PNG" exports at 2× (3200×1800). The bar heights are made up.</p>
{"".join(fig(*f) for f in FIGS)}
</main><script>
function svgOf(k){{return new XMLSerializer().serializeToString(document.querySelector('#'+k+' svg'))}}
function save(href,name){{const a=document.createElement('a');a.href=href;a.download=name;a.click()}}
function svgdl(k){{save(URL.createObjectURL(new Blob([svgOf(k)],{{type:'image/svg+xml'}})),'retrieval_hints_'+k+'.svg')}}
function png(k){{const img=new Image();img.onload=()=>{{const c=document.createElement('canvas');c.width=3200;c.height=1800;
c.getContext('2d').drawImage(img,0,0,3200,1800);c.toBlob(b=>save(URL.createObjectURL(b),'retrieval_hints_'+k+'.png'))}};
img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svgOf(k))}}
</script></body></html>'''
out = Path(os.environ["ART"]) / "visualize-how-it-works.html"
out.write_text(html)
print(out, out.stat().st_size)
