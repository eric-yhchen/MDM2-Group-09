#!/bin/bash
export GPG_TTY=$(tty)

echo -e "\nEnter the commit message:\n"
read commit_message

git add -A
git commit -m "$commit_message"
git push