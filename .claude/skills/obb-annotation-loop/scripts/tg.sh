#!/bin/bash
# Telegram progress helper. Credentials come from the environment so nothing secret lives in the repo:
#   export TG_BOT_TOKEN=...   export TG_CHATS="chat_id1 chat_id2"
# usage: tg.sh text "message" | tg.sh photo file.jpg "caption"
# Photos must stay under ~10 MB / 10000 px (width+height); make a 2x2 grid instead of a very wide strip - Telegram shrinks wide images to unreadable.
: "${TG_BOT_TOKEN:?set TG_BOT_TOKEN}"; : "${TG_CHATS:?set TG_CHATS}"
kind=$1; shift
if [ "$kind" = text ]; then body=$1; else file=$1; cap=$2; fi
for c in $TG_CHATS; do
  if [ "$kind" = text ]; then curl -s -o /dev/null -w "$c text %{http_code}\n" "https://api.telegram.org/bot$TG_BOT_TOKEN/sendMessage" -d chat_id=$c --data-urlencode text="$body"
  else curl -s -o /dev/null -w "$c photo %{http_code}\n" "https://api.telegram.org/bot$TG_BOT_TOKEN/sendPhoto" -F chat_id=$c -F photo=@"$file" -F caption="$cap"; fi
done
