# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class RobostrategyError(Exception):
    """A custom core Robostrategy exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(RobostrategyError, self).__init__(message)


class RobostrategyNotImplemented(RobostrategyError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(RobostrategyNotImplemented, self).__init__(message)


class RobostrategyAPIError(RobostrategyError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Robostrategy API'
        else:
            message = 'Http response error from Robostrategy API. {0}'.format(message)

        super(RobostrategyAPIError, self).__init__(message)


class RobostrategyApiAuthError(RobostrategyAPIError):
    """A custom exception for API authentication errors"""
    pass


class RobostrategyMissingDependency(RobostrategyError):
    """A custom exception for missing dependencies."""
    pass


class RobostrategyWarning(Warning):
    """Base warning for Robostrategy."""


class RobostrategyUserWarning(UserWarning, RobostrategyWarning):
    """The primary warning class."""
    pass


class RobostrategySkippedTestWarning(RobostrategyUserWarning):
    """A warning for when a test is skipped."""
    pass


class RobostrategyDeprecationWarning(RobostrategyUserWarning):
    """A warning for deprecated features."""
    pass
