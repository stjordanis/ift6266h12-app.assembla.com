#!/usr/bin/python
"""
Script that crawls the studium website and clones a course website.
For it to work you need to log in.
"""

import ConfigParser
import mechanize
import BeautifulSoup
import optparse
import os
import time
import sys


def grab_redirect(link):
    response = mechanize.urlopen(link['href'])
    forms = mechanize.ParseResponse(response, backwards_compat=False)
    form = forms[0]
    data = mechanize.urlopen(form.click()).read()
    soup = BeautifulSoup.BeautifulSoup(data)
    for div in soup('div'):
        if 'class' in dict(div.attrs) and \
           div['class'] == 'urlworkaround':
            txt = ''.join([str(x) for x in div.contents])
            lsoup = BeautifulSoup.BeautifulSoup(txt)
            link = lsoup('a')[0]
            return link['href']
    raise Exception('no href')


def start_cloning(options):
    link = options['link']
    user = options['user']
    password = options['password']
    response = mechanize.urlopen(link)
    forms = mechanize.ParseResponse(response, backwards_compat=False)
    form = forms[0]
    form['txtIdentifiant'] = user
    form['txtMDP'] = password
    website = mechanize.urlopen(form.click())
    data = website.read()
    outfile = open('index.html', 'wt')
    print >> outfile, """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html dir="ltr" lang="fr" xml:lang="fr" xmlns="http://www.w3.org/1999/xhtml"
class="yui3-js-enabled" id="yui_3_2_0_1_1326674808791714">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
"""
    soup = BeautifulSoup.BeautifulSoup(data)
    title = soup('title')
    print >>outfile, str(title[0])
    divs = soup('div')
    for div in divs:
        if 'class' in dict(div.attrs):
            if div['class'] == 'course-content':
                vstr = '\n'.join([str(x) for x in div.contents[1:]])
                # Eliminate wrong divs
                lsoup = BeautifulSoup.BeautifulSoup(vstr)
                for ldiv in lsoup.findAll('div'):
                    if ('class' in dict(ldiv.attrs) and
                        ldiv['class'] in ['left side', 'right side',
                                          'jumpmenu']):
                        ldiv.extract()
                replace = {}
                for link in lsoup.findAll('a'):
                    if 'href' in dict(link.attrs):
                        try:
                            replace[link['href']] = grab_redirect(link)
                        except:
                            pass
                page_txt = str(lsoup)
                for k, v in replace.items():
                    nw_key = str(k) +"&amp;redirect=1"
                    page_txt = page_txt.replace(nw_key, str(v))
                    page_txt = page_txt.replace(str(k), str(v))
                print >>outfile, page_txt
    outfile.close()


def main(parser, default='config.ini'):
    (options, args) = parser.parse_args()
    options = options.__dict__
    config = ConfigParser.ConfigParser()
    config.optionxform = str
    if 'configfile' in options and options['configfile'] is not None:
        config.readfp(open(options['configfile']))
    else:
        config.readfp(open(default))
    o = dict(config.items('global'))
    for k, v in o.iteritems():
        if k in options and options[k] is not None:
            o[k] = options[k]
        else:
            o[k] = v
    for k, v in options.iteritems():
        if k not in o:
            o[k] = v
    if 'link' not in o or o['link'] is None:
        print 'No link provided !'
        print
        parser.print_help()
        sys.exit(0)
    if 'user' not in o or o['user'] is None:
        print 'No user provided !'
        print
        parser.print_help()
        sys.exit(0)
    if 'password' not in o or o['password'] is None:
        print 'No password provided !'
        print
        parser.print_help()
        sys.exit(0)

    start_cloning(o)


def get_parser():
    usage = """
The script can read the configuration either from the config file, or these
values can be passed as arguments. Note that if they are both in the script
and passed as options to the script, the option passed value is used """
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-u',
                      "--user",
                      dest='user',
                      help=('User Name for logging on '
                            'studium.iro.umontreal.ca'))
    parser.add_option('-p',
                      "--password",
                      dest='password',
                      help=('Password for logging on '
                            'studium.iro.umontreal.ca'))
    parser.add_option('-l',
                      "--link",
                      dest='link',
                      help='Link to the class website')
    parser.add_option('-c',
                      "--config",
                      dest='configfile',
                      help='Config file for the script')
    parser.add_option('-v',
                      dest='verbose',
                      action='store_true',
                      help='Be verbose')
    return parser

if __name__ == '__main__':
    main(get_parser())
