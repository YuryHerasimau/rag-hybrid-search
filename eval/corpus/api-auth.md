---
title: Авторизация в API
---

# Авторизация в API

Запросы к API Орбиты авторизуются по access-токену. Токен передаётся в
заголовке `Authorization: Bearer <token>`.

## Получение токена

Метод `POST /api/v1/auth/token` принимает `email` и `password` и возвращает
`access_token` (живёт 1 час) и `refresh_token` (живёт 30 дней).

## Обновление токена

Когда access_token истёк, вызовите `POST /api/v1/auth/refresh` с refresh_token —
вернётся новая пара токенов. Повторное использование refresh_token запрещено.
