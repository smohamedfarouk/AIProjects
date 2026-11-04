#!/bin/bash
# The image grants MYSQL_USER full rights on MYSQL_DATABASE; the agent only needs to read.
# Defence in depth behind the agent's read-only SQL guardrail.
set -e
mysql -uroot -p"$MYSQL_ROOT_PASSWORD" <<SQL
REVOKE ALL PRIVILEGES ON \`$MYSQL_DATABASE\`.* FROM '$MYSQL_USER'@'%';
GRANT SELECT, SHOW VIEW ON \`$MYSQL_DATABASE\`.* TO '$MYSQL_USER'@'%';
FLUSH PRIVILEGES;
SQL
