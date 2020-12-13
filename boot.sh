#!/bin/sh
source venv/bin/activate
exec gunicorn -b :8400 --access-logfile - --error-logfile - app --timeout 300
