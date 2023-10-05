#!/bin/bash
ps aux | grep -E -- '[v]isualize|[d]efunct' | awk '{print($2)}' |xargs kill
