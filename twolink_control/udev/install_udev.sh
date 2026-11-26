#!/usr/bin/env bash
# 사용법: bash install_udev.sh   (sudo 비밀번호 필요)
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo cp "$DIR/99-twolink.rules" /etc/udev/rules.d/99-twolink.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
sleep 1
if [ -e /dev/twolink ]; then
  echo "OK: /dev/twolink -> $(readlink -f /dev/twolink)"
else
  echo "규칙 설치 완료. 로봇 USB 를 다시 꽂으면 /dev/twolink 가 생깁니다."
fi
