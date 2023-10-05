#!/bin/bash
ps aux | grep -E -- '[p]redict|[d]efunct' | awk '{print($2)}' |xargs kill -9
