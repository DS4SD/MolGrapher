#!/bin/bash
ps aux | grep -E -- '[t]rain_keypoint|[d]efunct' | awk '{print($2)}' |xargs kill -9
