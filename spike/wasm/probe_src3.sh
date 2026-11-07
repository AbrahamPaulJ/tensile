#!/bin/bash
cd ~/fe-master/ingots/core/src
grep -n "pub fn new\|pub fn try_new\|pub fn alloc\|pub fn ptr\|pub fn read_word\|pub fn write_word\|pub struct MemBuffer\|pub struct MemSpan\|pub fn span\|pub fn as_span" ptr.fe | head -30
