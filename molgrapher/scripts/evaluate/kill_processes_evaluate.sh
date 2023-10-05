#!/bin/bash
ps aux | grep -E -- '[e]valuate|[d]efunct' | awk '{print($2)}' |xargs kill
