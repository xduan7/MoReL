""" 
    File Name:          MoReL/strict_typing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/1/19
    Python Version:     3.5.4
    File Description:   
        https://stackoverflow.com/questions/32684720/
        how-do-i-ensure-parameter-is-correct-type-in-python
"""
from typing import get_type_hints
from deprecated import deprecated


@deprecated(reason='This type checking does not perform base class checking.')
def strict_typing(f):
    def type_checker(*args, **kwargs):

        hints = get_type_hints(f)

        all_args = kwargs.copy()
        all_args.update(dict(zip(f.__code__.co_varnames, args)))

        for key in all_args:
            if key in hints:
                __hint = hints[key]
                if type(all_args[key]) != hints[key]:
                    raise TypeError(f'Type of {key} is {type(all_args[key])} '
                                    f'and not {hints[key]}')

        return f(*args, **kwargs)
    return type_checker
