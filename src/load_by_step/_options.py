#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__all__ = [
    "set_options",
    "get_options",
]


OPTIONS = dict(
    tqdm_disable=False,
)


class set_options:

    def __init__(self,
                 tqdm_disable: bool = False,
                 ) -> None:

        OPTIONS["tqdm_disable"] = tqdm_disable


def get_options():
    return OPTIONS.copy()
