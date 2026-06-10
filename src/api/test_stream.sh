#!/bin/bash

curl -N -X POST http://127.0.0.1:8000/runs/stream \
     -H "Content-Type: application/json" \
     -d '{
       "question": "PMI项目是啥",
       "thread_id": "curl-test-session"
     }'
