"""Progress v009: unchanged v008 data feed with responsive UI v002."""
import argparse
from pathlib import Path
import server_v008 as original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    original.UI = Path(__file__).with_name('progress_v002.html')
    print(f'Live project progress v009: http://127.0.0.1:{args.port}', flush=True)
    original.ThreadingHTTPServer(('127.0.0.1', args.port), original.Handler).serve_forever()
