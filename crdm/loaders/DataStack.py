from datetime import datetime as dt
from datetime import timedelta

date_str = os.path.basename(target.as_posix())[:8]
date = dt.strptime(date_str, '%Y%M%d').date()
date_range = [str(date - timedelta(weeks=x)).replace('-', '') for x in range(1, 9)]
mon_range = set([x[:6] for x in date_range])

weekly_candidates = [x + '_' + y + '.dat' for x in date_range for y in WEEKLY_VARS]