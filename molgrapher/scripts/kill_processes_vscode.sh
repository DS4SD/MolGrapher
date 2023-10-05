#!/bin/bash
ps aux | grep -E -- '[v]scode' | awk '{print($2)}' |xargs kill
