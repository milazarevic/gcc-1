/* Main for jv-scan
   Copyright (C) 1998 Free Software Foundation, Inc.
   Contributed by Alexandre Petit-Bianco (apbianco@cygnus.com)

This file is part of GNU CC.

GNU CC is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GNU CC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU CC; see the file COPYING.  If not, write to
the Free Software Foundation, 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.  */

#include <stdio.h>
#include <stdlib.h>
#ifdef __STDC__
#include <stdarg.h>
#else
#include <varargs.h>
#endif

#include "gansidecl.h"		/* Definitions of PROTO and VPROTO */
#include "obstack.h"		/* We use obstacks in lex.c */

void fatal VPROTO((char *s, ...));
void warning VPROTO((char *s, ...));
void gcc_obstack_init PROTO ((struct obstack *obstack));
extern void reset_report PROTO ((void));

#define JC1_LITE
#include "parse.h"

/* Current input file and output file IO streams.  */
FILE *finput, *out;

/* Current input filename.  */
char *input_filename;

/* Executable name.  */
char *exec_name;

/* Flags matching command line options.  */
int flag_find_main = 0;
int flag_dump_class = 0;
int flag_list_filename = 0;

/* jc1-lite main entry point */
int
main (argc, argv, envp)
     int argc;
     char **argv;
     char **envp;
{
  int i = 1;
  char *output_file = NULL;
  long ft;

  exec_name = argv[0];

  /* Default for output */
  out = stdout;

  /* Process options first */
  while (argv [i])
    {
      if (argv [i][0] == '-')
	{
	  /* Dump result into a file */
	  if (!strcmp (argv [i], "-o") && i+1 < argc)
	    {
	      argv [i] = NULL;
	      output_file = argv [++i];
	      argv [i] = NULL;
	    }

	  /* Print the name of the class that contains main */
	  else if (!strcmp (argv [i], "--print-main"))
	    flag_find_main = 1;

	  else if (!strcmp (argv [i], "--list-filename"))
	    flag_list_filename = 1;

	  /* List all the classes found in a source file */
	  else if (!strcmp (argv [i], "--list-class"))
	    flag_dump_class = 1;

	  else
	    warning ("Unrecognized argument `%s'", argv[i]);

	  /* non recognized argument ignored silently */ 
	  argv [i] = NULL;	/* Nullify so it's not considered a file */
	}
      i++;
    }

  /* No flags? Do nothing */
  if (!flag_find_main && !flag_dump_class)
    exit (0);

  /* Check on bad usage */
  if (flag_find_main && flag_dump_class)
    fatal ("Options `--print-main' and `--list-class' can't be turned on "
	   "at the same time", argv [0]);

  if (output_file && !(out = fopen (output_file, "w")))
    fatal ("Can't open output file `%s'", argv [0], output_file);

  ft = ftell (out);

  gcc_obstack_init (&temporary_obstack);
  java_push_parser_context ();

  for ( i = 1; i < argc; i++ )
    if (argv [i])
      {
	input_filename = argv [i];
	if ( (finput = fopen (argv [i], "r")) )
	  {
	    java_init_lex ();
	    yyparse ();
	    if (ftell (out) != ft)
	      fputc ('\n', out);
	    ft = ftell (out);
	    fclose (finput);
	    reset_report ();
	  }
	else
	  fatal ("File not found `%s'", argv [0], argv [i]);
      }

  /* Flush and close */
  if (ftell (out) != ft)
    fputc ('\n', out);
  if (!output_file)
    fclose (out);

  exit (0);
}

/* Error report, memory, obstack initialization and other utility
   functions */

void
fatal VPROTO((char *s, ...))
{
#ifndef __STDC__
  char *s;
#endif
  va_list ap;

  VA_START (ap, s);

#ifndef __STDC__
  s = va_arg (ap, char *);
#endif

  fprintf (stderr, "%s: error: ", exec_name);
  vfprintf (stderr, s, ap);
  fputc ('\n', stderr);
  va_end (ap);
  exit (1);
}

char *
xmalloc (size)
     unsigned size;
{
  register char *value;

  if (size == 0)
    size = 1;

  value = (char *) malloc (size);
  if (value == 0)
    fatal ("virtual memory exhausted");
  return value;
}

char *
xstrdup (string)
     char *string;
{
  int length = strlen (string)+1;
  char *to_return = xmalloc (length);
  strcpy (to_return, string);
  return to_return;
}

void
warning VPROTO((char *s, ...))
{
#ifndef __STDC__
  char *s;
#endif
  va_list ap;

  VA_START (ap, s);

#ifndef __STDC__
  s = va_arg (ap, char *);
#endif

  fprintf (stderr, "%s: warning: ", exec_name);
  vfprintf (stderr, s, ap);
  fputc ('\n', stderr);
  va_end (ap);
}

void
gcc_obstack_init (obstack)
     struct obstack *obstack;
{
  /* Let particular systems override the size of a chunk.  */
#ifndef OBSTACK_CHUNK_SIZE
#define OBSTACK_CHUNK_SIZE 0
#endif
  /* Let them override the alloc and free routines too.  */
#ifndef OBSTACK_CHUNK_ALLOC
#define OBSTACK_CHUNK_ALLOC xmalloc
#endif
#ifndef OBSTACK_CHUNK_FREE
#define OBSTACK_CHUNK_FREE free
#endif
  _obstack_begin (obstack, OBSTACK_CHUNK_SIZE, 0,
		  (void *(*) ()) OBSTACK_CHUNK_ALLOC,
		  (void (*) ()) OBSTACK_CHUNK_FREE);
}
