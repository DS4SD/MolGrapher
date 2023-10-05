#!/bin/bash
ps aux | grep -E -- '[t]rain_graph|[d]efunct' | awk '{print($2)}' |xargs kill -9
