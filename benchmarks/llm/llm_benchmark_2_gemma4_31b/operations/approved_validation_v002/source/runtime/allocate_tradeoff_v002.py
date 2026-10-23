"""Single authorized allocation with sanitized HTML/plain service diagnostics."""
from html.parser import HTMLParser
import json
import re
import sys
import colab_control_v003 as v6
import colab_control_v002 as v5

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

if __name__=='__main__':
    hardware=sys.argv.pop(1)
    if hardware not in ('v5e','v6e'):raise ValueError('Explicit supported hardware required')
    control=v6 if hardware=='v6e' else v5
    try:
        raise SystemExit(control.main())
    except Exception as exc:
        record={'kind':'allocation_error','hardware':hardware,**control.safe_error(exc)}
        request=getattr(exc,'request',None)
        if request is not None:
            record['method']=request.method
            for key in ('Authorization','X-Goog-Colab-Token'):
                control.remember_secret(request.headers.get(key))
        body=getattr(exc,'response_body','') or ''
        try:record['service_error']=json.loads(body.removeprefix(")]}'\n"))
        except ValueError:
            parser=Text();parser.feed(body)
            message=' '.join(' '.join(parser.parts).split())
            message=re.sub(r'https?://\S+','[URL omitted]',message)
            record['service_message']=control.safe_text(message)[:1200]
        control.emit(record)
        raise SystemExit(1)
