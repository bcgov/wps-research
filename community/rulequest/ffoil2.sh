
# This is a shell archive.  Remove anything before this line,
# then unpack it by saving it in a file and typing "sh file".
#
# Wrapped by quinlan on Fri Sep 27 09:29:52 EST 1996
# Contents:  Release2/ Release2/Examples/ Release2/Src/ Release2/MANUAL.foil6
#	Release2/README Release2/Examples/ackermann.d Release2/Examples/gcd.d
#	Release2/Examples/mesh_a.d Release2/Examples/mesh_b.d
#	Release2/Examples/mesh_c.d Release2/Examples/mesh_d.d
#	Release2/Examples/mesh_e.d Release2/Examples/qs44.d
#	Release2/Examples/sort.d Release2/Src/Makefile Release2/Src/constants.c
#	Release2/Src/defns.i Release2/Src/determinate.c
#	Release2/Src/evaluatelit.c Release2/Src/extern.i Release2/Src/finddef.c
#	Release2/Src/global.c Release2/Src/input.c Release2/Src/interpret.c
#	Release2/Src/join.c Release2/Src/literal.c Release2/Src/main.c
#	Release2/Src/order.c Release2/Src/output.c Release2/Src/prune.c
#	Release2/Src/search.c Release2/Src/state.c Release2/Src/utility.c
 
echo mkdir - Release2
mkdir Release2
chmod u=rwx,g=,o= Release2
 
echo mkdir - Release2/Examples
mkdir Release2/Examples
chmod u=rwx,g=rx,o=rx Release2/Examples
 
echo x - Release2/MANUAL.foil6
sed 's/^@//' > "Release2/MANUAL.foil6" <<'@//E*O*F Release2/MANUAL.foil6//'
NAME
	foil6 - produce Horn clauses from relational data

SYNOPSIS
	foil6 [ -n ] [ -N ] [ -v verb ] [ -V vars ] [ -s frac ] [ -m maxt ]
	      [ -d depth ] [ -w weaklits ] [ -a accur ] [ -l alter ]
	      [ -t chkpt ] [ -f gain ] [ -g max ]

DESCRIPTION
	FOIL is a program that reads extensional specifications of a set of
	relations and produces Horn clause definitions of one or more of them.

INPUT
*****

Input to the program consists of three sections:

        * specification of types
          blank line
        * extensional definitions of relations
          blank line                            |  these are
        * test cases for learned definitions    |  optional

Types
-----

Each discrete type specification consists of the type name followed by a colon,
then a series of constants separated by commas and terminated with a period.
This may occupy several lines and the same constant can appear in many types.

There are three kinds of discrete types:

	* ordered types (type name preceded by '*')
	  The constants have a natural order and appear in this order in
	  the type definition, smallest constant first.
	* unordered types (type name preceded by '#')
	  The constants do not have any natural order.
	* possibly ordered types
	  FOIL will attempt to discover an ordering of the constants that
	  may be useful for recursive definitions.

Each continuous type specification consists of the type name followed by
": continuous." on one line.  The constants corresponding to a continuous type
are the usual integers and real numbers -- any string that can be converted to
a float by C's atof() should work when specifying a value in a tuple.

Constants
---------

A non-numeric constant consist of any string of characters with the exception
that any occurrence of certain delimiter characters (left and right parenthesis,
period, comma, semicolon) must be prefixed by the escape character '\'.
A "theory" constant that can appear in a definition should be preceded by
'*'.  Two one-character constants have a special meaning and should not
be used otherwise:

	* '?' indicates a missing or unknown value
	      (see Cameron-Jones and Quinlan, 1993b)
	* '^' indicates an out-of-closed-world value
	      (see Quinlan and Cameron-Jones, 1994)

Relations
---------

All relations are defined in terms of the set of positive tuples of constants
for which the relation is true, and optionally the set of negative tuples of
constants for which it is false.  If only positive tuples are given, all other
constant tuples of the correct types are considered to be negative.

Each relation is defined by a header and one or two sets of constant tuples.
The header can be specified as follows:

    name(type, type, ... , type) key/key/.../key

The header of all relations other than target relations begins with '*'.
The header consists of relation name, argument types and optional keys.
Keys limit the ways the relation may be used and consist of one character
for each type.  The character '#' indicates that the corresponding argument in
a literal must be bound; the character '-' indicates that the argument can be
bound or unbound.  Each key thus gives a permissible way of accessing the
relation.  If no keys appear, all possible combinations of bound and unbound
arguments are allowed.

Following the header line are a series of lines containing constant tuples:

    positive tuple
    positive tuple
      . . .
    ;			| these
    negative tuple	| are
    negative tuple	| optional
      . . .		|
    .

Each tuple consists of constants separated by commas and must appear on a
single line.  The character ';' separates positive tuples from negative
tuples, which are optional.

Tests
-----

The optional test relations may be given to test the learned Horn clause 
definitions.  The additional input consists of

        a blank line (indicating start of test relation specification)
        relation name
        test tuples
        .
        relation name
        test tuples
        .
          and so on

Each test tuple consists of a constant tuple followed by ": +" if it is 
belongs to the relation and ": -" if it does not.  The definition interpreter is
simple; right-hand sides of the clauses are checked with reference to the
given tuples, not to the definitions of the relations that may have been
learned.

OPTIONS
*******

Options and their meanings are:

        -n      Negative literals are not considered.  This may be useful in
                domains where negated literals wouldn't make sense, or if
                learned definitions must be Horn clauses.

	-N	This is similar, but permits negated equality literals
		A<>B and A<>constant.

        -vverb	Set verbosity level [0, 1, 2, 3, or 4; default 1]
                The program produces rather voluminous trace output controlled
                by this variable.  The default value of 1 gives a fair
                amount of detail; 0 produces very little output; 3 gives
                a blow-by-blow account of what the system is doing;
                4 gives details of tuples in training sets etc.

	-Vvars	Set the maximum number of variables that can be used during
		the search for a definition. [default: 52]

        -sfrac	In some predicates of high arity, the closed world assumption
                will generate very many negative tuples.  This option causes
                only a randomly-selected neg% of negative tuples to be used.
                Note that this option has no effect if negative tuples are
                given explicitly.

	-mmaxt	Set the maximum number of tuples; the default is 100000.
		If the default setting results in warnings that literals are
		being excluded due to the tuple limit, expanding the limit
		may be useful (but time-consuming).

        -ddepth	Set the maximum variable depth [default 4].  This limits the
                possible depth of variables in literals.

	-wwklts Set the maximum number of weak (zero-gain) literals that
		can appear in sequence [default: 4].  A batch of determinate
		literals counts as one literal in this respect.

        -aaccur	Set the minimum accuracy of any clause [default 80%]
                FOIL will not accept any clause with an accuracy lower
                than this.

	-lalter Set the maximum number of alternatives to any literal
		[default 5].  This limits the amount of backup from any 
		one point.

        -tchkpt	Set the maximum number of checkpoints at any one time 
		[default 20].

        -fgain	Any alternative literal must have at least gain%
                of the best literal gain [default 80%].  

        -gmax	Determinate literals are automatically included, unless
                there is a literal which has at least max% of the maximum
                possible gain.  (The maximum possible gain is achieved
                by a literal that is satisfied by all + tuples, but no
                - tuples, in the current training set.)  Obviously, if
                max is zero, no determinate literals are included unless
                there are no other literals.


SEE ALSO

	Quinlan, J.R. (1990), "Learning Logical Definitions from Relations",
	Machine Learning 5, 239-266.

	Quinlan, J.R. (1991), "Determinate Literals in Inductive Logic
	Programming", Proceedings 12th International Joint Conference on
	Artificial Intelligence, 746-750, Morgan Kaufmann.

	Quinlan, J.R. and Cameron-Jones, R.M. (1993), "FOIL: a midterm report",
	3-20, Proceedings European Conference on Machine Learning, Springer
	Verlag.

	Cameron-Jones, R.M. and Quinlan, J.R. (1993a), "Avoiding Pitfalls When
	Learning Recursive Theories", Proceedings IJCAI 93, 1050-1055,
	Morgan Kaufmann.

	Cameron-Jones, R.M. and Quinlan, J.R., (1993b), "First Order Learning,
	Zeroth Order Data", Sixth Australian Joint Conference on Artificial
	Intelligence, World Scientific.

	Quinlan, J.R. and Cameron-Jones, R.M., (1994), "Living in a Closed
	World", draft available by anonymous ftp from ftp.cs.su.oz.au
	(file pub/q+cj.closed.ps).
@//E*O*F Release2/MANUAL.foil6//
chmod u=rw,g=r,o=r Release2/MANUAL.foil6
 
echo x - Release2/README
sed 's/^@//' > "Release2/README" <<'@//E*O*F Release2/README//'
FFOIL 1.0
---------

This program is based on FOIL release 6.3.  Input is very similar --
I haven't written a manual yet, but the FOIL6 manual (copied here)
should suffice.  The principal differences are
  *  the target relation must be functional, the last constant in each
     tuple being the value of the function
  *  negative examples are not needed (and should not be specified!)
  *  a couple of FOIL6 options to do with negative examples (e.g. -s)
     are no longer relevant
  *  the -s option now has no argument and means "do not carry out
     global clause simplification".  ffoil2 with the -s option
     behaves similarly to ffoil1.
Several sample files are provided with file names ending in .d;
for these examples, I recommend using the -n option with all but
the past tense task files (ph----) for which -N is more appropriate.

This is the first release of FFOIL, so I would appreciate reports on
any bugs encountered (to quinlan@cs.su.oz.au).

Ross
@//E*O*F Release2/README//
chmod u=rw,g=r,o=r Release2/README
 
echo mkdir - Release2/Src
mkdir Release2/Src
chmod u=rwx,g=rx,o=rx Release2/Src
 
echo x - Release2/Examples/ackermann.d
sed 's/^@//' > "Release2/Examples/ackermann.d" <<'@//E*O*F Release2/Examples/ackermann.d//'
*N: *0,*1,2,3,4,5,6,7,8,9,10,
   11,12,13,14,15,16,17,18,19,20.

Ackermann(N,N,N) ##-
0,0,1
0,1,2
0,2,3
0,3,4
0,4,5
0,5,6
0,6,7
0,7,8
0,8,9
0,9,10
0,10,11
0,11,12
0,12,13
0,13,14
0,14,15
0,15,16
0,16,17
0,17,18
0,18,19
0,19,20
1,0,2
1,1,3
1,2,4
1,3,5
1,4,6
1,5,7
1,6,8
1,7,9
1,8,10
1,9,11
1,10,12
1,11,13
1,12,14
1,13,15
1,14,16
1,15,17
1,16,18
1,17,19
1,18,20
2,0,3
2,1,5
2,2,7
2,3,9
2,4,11
2,5,13
2,6,15
2,7,17
2,8,19
3,0,5
3,1,13
4,0,13
@.
*succ(N,N)
0,1
1,2
2,3
3,4
4,5
5,6
6,7
7,8
8,9
9,10
10,11
11,12
12,13
13,14
14,15
15,16
16,17
17,18
18,19
19,20
@.
@//E*O*F Release2/Examples/ackermann.d//
chmod u=r,g=r,o=r Release2/Examples/ackermann.d
 
echo x - Release2/Examples/gcd.d
sed 's/^@//' > "Release2/Examples/gcd.d" <<'@//E*O*F Release2/Examples/gcd.d//'
*I: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.

*plus(I,I,I)
1,1,2
1,2,3
1,3,4
1,4,5
1,5,6
1,6,7
1,7,8
1,8,9
1,9,10
1,10,11
1,11,12
1,12,13
1,13,14
1,14,15
1,15,16
1,16,17
1,17,18
1,18,19
1,19,20
2,1,3
2,2,4
2,3,5
2,4,6
2,5,7
2,6,8
2,7,9
2,8,10
2,9,11
2,10,12
2,11,13
2,12,14
2,13,15
2,14,16
2,15,17
2,16,18
2,17,19
2,18,20
3,1,4
3,2,5
3,3,6
3,4,7
3,5,8
3,6,9
3,7,10
3,8,11
3,9,12
3,10,13
3,11,14
3,12,15
3,13,16
3,14,17
3,15,18
3,16,19
3,17,20
4,1,5
4,2,6
4,3,7
4,4,8
4,5,9
4,6,10
4,7,11
4,8,12
4,9,13
4,10,14
4,11,15
4,12,16
4,13,17
4,14,18
4,15,19
4,16,20
5,1,6
5,2,7
5,3,8
5,4,9
5,5,10
5,6,11
5,7,12
5,8,13
5,9,14
5,10,15
5,11,16
5,12,17
5,13,18
5,14,19
5,15,20
6,1,7
6,2,8
6,3,9
6,4,10
6,5,11
6,6,12
6,7,13
6,8,14
6,9,15
6,10,16
6,11,17
6,12,18
6,13,19
6,14,20
7,1,8
7,2,9
7,3,10
7,4,11
7,5,12
7,6,13
7,7,14
7,8,15
7,9,16
7,10,17
7,11,18
7,12,19
7,13,20
8,1,9
8,2,10
8,3,11
8,4,12
8,5,13
8,6,14
8,7,15
8,8,16
8,9,17
8,10,18
8,11,19
8,12,20
9,1,10
9,2,11
9,3,12
9,4,13
9,5,14
9,6,15
9,7,16
9,8,17
9,9,18
9,10,19
9,11,20
10,1,11
10,2,12
10,3,13
10,4,14
10,5,15
10,6,16
10,7,17
10,8,18
10,9,19
10,10,20
11,1,12
11,2,13
11,3,14
11,4,15
11,5,16
11,6,17
11,7,18
11,8,19
11,9,20
12,1,13
12,2,14
12,3,15
12,4,16
12,5,17
12,6,18
12,7,19
12,8,20
13,1,14
13,2,15
13,3,16
13,4,17
13,5,18
13,6,19
13,7,20
14,1,15
14,2,16
14,3,17
14,4,18
14,5,19
14,6,20
15,1,16
15,2,17
15,3,18
15,4,19
15,5,20
16,1,17
16,2,18
16,3,19
16,4,20
17,1,18
17,2,19
17,3,20
18,1,19
18,2,20
19,1,20
@.
gcd(I,I,I) ##-
1,1,1
1,2,1
1,3,1
1,4,1
1,5,1
1,6,1
1,7,1
1,8,1
1,9,1
1,10,1
1,11,1
1,12,1
1,13,1
1,14,1
1,15,1
1,16,1
1,17,1
1,18,1
1,19,1
1,20,1
2,1,1
2,2,2
2,3,1
2,4,2
2,5,1
2,6,2
2,7,1
2,8,2
2,9,1
2,10,2
2,11,1
2,12,2
2,13,1
2,14,2
2,15,1
2,16,2
2,17,1
2,18,2
2,19,1
2,20,2
3,1,1
3,2,1
3,3,3
3,4,1
3,5,1
3,6,3
3,7,1
3,8,1
3,9,3
3,10,1
3,11,1
3,12,3
3,13,1
3,14,1
3,15,3
3,16,1
3,17,1
3,18,3
3,19,1
3,20,1
4,1,1
4,2,2
4,3,1
4,4,4
4,5,1
4,6,2
4,7,1
4,8,4
4,9,1
4,10,2
4,11,1
4,12,4
4,13,1
4,14,2
4,15,1
4,16,4
4,17,1
4,18,2
4,19,1
4,20,4
5,1,1
5,2,1
5,3,1
5,4,1
5,5,5
5,6,1
5,7,1
5,8,1
5,9,1
5,10,5
5,11,1
5,12,1
5,13,1
5,14,1
5,15,5
5,16,1
5,17,1
5,18,1
5,19,1
5,20,5
6,1,1
6,2,2
6,3,3
6,4,2
6,5,1
6,6,6
6,7,1
6,8,2
6,9,3
6,10,2
6,11,1
6,12,6
6,13,1
6,14,2
6,15,3
6,16,2
6,17,1
6,18,6
6,19,1
6,20,2
7,1,1
7,2,1
7,3,1
7,4,1
7,5,1
7,6,1
7,7,7
7,8,1
7,9,1
7,10,1
7,11,1
7,12,1
7,13,1
7,14,7
7,15,1
7,16,1
7,17,1
7,18,1
7,19,1
7,20,1
8,1,1
8,2,2
8,3,1
8,4,4
8,5,1
8,6,2
8,7,1
8,8,8
8,9,1
8,10,2
8,11,1
8,12,4
8,13,1
8,14,2
8,15,1
8,16,8
8,17,1
8,18,2
8,19,1
8,20,4
9,1,1
9,2,1
9,3,3
9,4,1
9,5,1
9,6,3
9,7,1
9,8,1
9,9,9
9,10,1
9,11,1
9,12,3
9,13,1
9,14,1
9,15,3
9,16,1
9,17,1
9,18,9
9,19,1
9,20,1
10,1,1
10,2,2
10,3,1
10,4,2
10,5,5
10,6,2
10,7,1
10,8,2
10,9,1
10,10,10
10,11,1
10,12,2
10,13,1
10,14,2
10,15,5
10,16,2
10,17,1
10,18,2
10,19,1
10,20,10
11,1,1
11,2,1
11,3,1
11,4,1
11,5,1
11,6,1
11,7,1
11,8,1
11,9,1
11,10,1
11,11,11
11,12,1
11,13,1
11,14,1
11,15,1
11,16,1
11,17,1
11,18,1
11,19,1
11,20,1
12,1,1
12,2,2
12,3,3
12,4,4
12,5,1
12,6,6
12,7,1
12,8,4
12,9,3
12,10,2
12,11,1
12,12,12
12,13,1
12,14,2
12,15,3
12,16,4
12,17,1
12,18,6
12,19,1
12,20,4
13,1,1
13,2,1
13,3,1
13,4,1
13,5,1
13,6,1
13,7,1
13,8,1
13,9,1
13,10,1
13,11,1
13,12,1
13,13,13
13,14,1
13,15,1
13,16,1
13,17,1
13,18,1
13,19,1
13,20,1
14,1,1
14,2,2
14,3,1
14,4,2
14,5,1
14,6,2
14,7,7
14,8,2
14,9,1
14,10,2
14,11,1
14,12,2
14,13,1
14,14,14
14,15,1
14,16,2
14,17,1
14,18,2
14,19,1
14,20,2
15,1,1
15,2,1
15,3,3
15,4,1
15,5,5
15,6,3
15,7,1
15,8,1
15,9,3
15,10,5
15,11,1
15,12,3
15,13,1
15,14,1
15,15,15
15,16,1
15,17,1
15,18,3
15,19,1
15,20,5
16,1,1
16,2,2
16,3,1
16,4,4
16,5,1
16,6,2
16,7,1
16,8,8
16,9,1
16,10,2
16,11,1
16,12,4
16,13,1
16,14,2
16,15,1
16,16,16
16,17,1
16,18,2
16,19,1
16,20,4
17,1,1
17,2,1
17,3,1
17,4,1
17,5,1
17,6,1
17,7,1
17,8,1
17,9,1
17,10,1
17,11,1
17,12,1
17,13,1
17,14,1
17,15,1
17,16,1
17,17,17
17,18,1
17,19,1
17,20,1
18,1,1
18,2,2
18,3,3
18,4,2
18,5,1
18,6,6
18,7,1
18,8,2
18,9,9
18,10,2
18,11,1
18,12,6
18,13,1
18,14,2
18,15,3
18,16,2
18,17,1
18,18,18
18,19,1
18,20,2
19,1,1
19,2,1
19,3,1
19,4,1
19,5,1
19,6,1
19,7,1
19,8,1
19,9,1
19,10,1
19,11,1
19,12,1
19,13,1
19,14,1
19,15,1
19,16,1
19,17,1
19,18,1
19,19,19
19,20,1
20,1,1
20,2,2
20,3,1
20,4,4
20,5,5
20,6,2
20,7,1
20,8,4
20,9,1
20,10,10
20,11,1
20,12,4
20,13,1
20,14,2
20,15,5
20,16,4
20,17,1
20,18,2
20,19,1
20,20,20
@.
@//E*O*F Release2/Examples/gcd.d//
chmod u=r,g=r,o=r Release2/Examples/gcd.d
 
echo x - Release2/Examples/mesh_a.d
sed 's/^@//' > "Release2/Examples/mesh_a.d" <<'@//E*O*F Release2/Examples/mesh_a.d//'
#Elt:
 a1, a2, a3, a4, a5, a6, a7, a8, a9,a10,
a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,
a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,
a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,
a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,
a51,a52,a53,a54,a55,
 b1, b2, b3, b4, b5, b6, b7, b8, b9,b10,
b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,
b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,
b41,b42,
 c1, c2, c3, c4, c5, c6, c7, c8, c9,c10,
c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
c21,c22,c23,c24,c25,c26,c27,c28,
 d1, d2, d3, d4, d5, d6, d7, d8, d9,d10,
d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,
d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,
d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,
d51,d52,d53,d54,d55,d56,d57,
 e1, e2, e3, e4, e5, e6, e7, e8, e9,e10,
e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,
e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,
e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,
e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,
e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,
e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,
e81,e82,e83,e84,e85,e86,e87,e88,e89,e90,
e91,e92,e93,e94,e95,e96.
No:
*0,*1,*2,*3,*4,*5,*6,*7,*8,*9,*10,*11,*12.

*long(Elt)
a1
a34
a54
b19
b39
e19
e22
@.
*usual(Elt)
a3
a39
b11
b13
b15
b24
b25
b27
b31
b32
c5
c6
c8
c10
c12
c14
d1
d2
d28
d29
e2
e4
e5
e6
e7
e8
e14
e15
e16
e17
e18
e20
e21
e24
e25
e26
e27
e28
e29
e32
e34
e35
e37
e40
e43
e44
e45
e46
e48
e49
e50
e52
e55
e57
e58
e60
e62
e63
e65
e66
e67
e69
e70
e71
e72
e73
e74
e80
e81
e83
e86
e88
e89
e90
e91
e92
e93
e94
e95
@.
*short(Elt)
a6
a9
a11
a13
a15
a19
a22
a25
a26
a28
a31
a35
a40
a44
b5
b7
b8
b10
b16
b18
b26
b33
b36
b37
c7
c13
d3
d4
d6
d8
d10
d12
d14
d15
d16
d18
d20
d22
d24
d26
d27
e1
e3
e9
e12
e23
e30
e31
e33
e36
e38
e51
e53
e54
e56
e59
e61
e64
e68
e82
e87
@.
*circuit(Elt)
c15
c16
c17
c18
c19
c24
c25
c26
c27
c28
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
@.
*half_circuit(Elt)
a36
a37
a45
a46
a47
a48
a49
a50
a51
a52
a53
b3
b6
b9
b12
b17
b20
b41
b42
@.
*quarter_circuit(Elt)
e75
e76
e77
e78
e84
e85
@.
*short_for_hole(Elt)
a16
a17
a18
a23
a33
b28
b30
b34
b35
c3
c4
e13
e96
@.
*long_for_hole(Elt)
c2
e41
e79
@.
*circuit_hole(Elt)
c20
c21
c22
c23
@.
*half_circuit_hole(Elt)
a38
a42
a43
a55
b1
b14
b22
b29
b38
b40
e10
e11
e39
e47
@.
*not_important(Elt)
a2
a4
a5
a7
a8
a10
a12
a14
a20
a21
a24
a27
a29
a30
a32
a41
b2
b4
b21
b23
c1
c9
c11
d5
d7
d9
d11
d13
d17
d19
d21
d23
d25
e42
@.
*free(Elt)
a39
a40
c6
c7
c11
c12
c13
c15
c16
c17
c18
c19
c20
c27
c28
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d17
d18
d19
d20
d21
d22
d23
d24
d25
d26
d27
d29
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
e6
e16
e19
e20
e22
e23
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e36
e45
e48
e49
e51
e52
e53
e54
e66
e71
e73
e83
e86
e88
e90
e92
e93
e94
@.
*one_side_fixed(Elt)
a34
a35
a41
c2
c3
c5
c8
c10
c14
e5
e7
e15
e17
e18
e21
e24
e37
e44
e46
e50
e65
e67
e69
e74
e76
e77
e78
e82
e84
e85
e89
e91
e95
@.
*two_side_fixed(Elt)
a36
a37
a38
a42
a43
a45
a46
a47
a48
a49
a50
a51
a52
a53
a55
b6
b9
b12
b14
b17
b20
b22
b29
b38
b40
b41
b42
e10
e11
e39
e47
@.
*fixed(Elt)
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24
a25
a26
a27
a28
a29
a30
a31
a32
a33
a44
a54
b1
b2
b3
b4
b5
b7
b8
b10
b11
b13
b15
b16
b18
b19
b21
b23
b24
b25
b26
b27
b28
b30
b31
b32
b33
b34
b35
b36
b37
b39
c1
c4
c9
c21
c22
c23
c24
c25
c26
d1
d2
d28
d30
e1
e2
e3
e4
e8
e9
e12
e13
e14
e25
e38
e40
e41
e42
e43
e55
e56
e57
e58
e59
e60
e61
e62
e63
e64
e68
e70
e72
e75
e79
e80
e81
e87
e96
@.
*not_loaded(Elt)
a1
a2
a3
a4
a5
a6
a7
a23
a24
a25
a26
a27
a28
a29
a33
a36
a37
a42
a44
a45
a46
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12
b13
b14
b15
b16
b17
b18
b19
b20
b21
b23
b24
c1
c2
c3
c4
c5
c6
c7
c8
c9
c15
c17
c18
c20
c21
c22
c23
c26
d1
d2
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d20
d21
d22
d23
d24
d25
d26
d27
d28
d29
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d49
d50
d51
d52
d53
d54
d55
d56
d57
e1
e2
e3
e4
e5
e6
e7
e8
e9
e10
e11
e12
e13
e14
e15
e16
e17
e18
e19
e20
e21
e22
e23
e24
e25
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e39
e40
e41
e42
e43
e44
e45
e46
e47
e48
e49
e50
e51
e52
e53
e54
e55
e56
e57
e58
e62
e63
e64
e65
e66
e67
e68
e69
e70
e71
e72
e73
e74
e77
e78
e79
e80
e81
e82
e83
e84
e85
e86
e87
e88
e89
e90
e91
e92
e93
e94
e95
e96
@.
*one_side_loaded(Elt)
a34
a35
a40
a41
a54
d45
d46
d47
d48
e36
e38
e59
e61
@.
*two_side_loaded(Elt)
e37
@.
*cont_loaded(Elt)
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a30
a31
a32
a38
a39
a43
a47
a48
a49
a50
a51
a52
a53
b22
b25
b26
b27
b28
b29
b30
b31
b32
b33
b34
b35
b36
b37
b38
b39
b40
b41
b42
c10
c11
c12
c13
c14
c16
c19
c24
c25
c27
c28
d17
d18
d19
e60
e75
e76
@.
*neighbour(Elt,Elt)
a1,a2
a1,a44
a10,a11
a10,a49
a11,a12
a11,a49
a12,a13
a12,a51
a13,a14
a14,a15
a14,a53
a15,a16
a15,a38
a16,a17
a16,a55
a17,a18
a18,a19
a18,a38
a19,a20
a2,a3
a2,a54
a20,a21
a20,a52
a21,a22
a22,a23
a22,a50
a23,a24
a24,a1
a24,a42
a25,a26
a25,a35
a26,a27
a26,a36
a27,a28
a27,a37
a28,a29
a28,a37
a29,a30
a29,a46
a3,a4
a30,a31
a30,a46
a31,a32
a31,a48
a32,a33
a32,a48
a33,a25
a34,a35
a34,a40
a35,a26
a35,a40
a36,a27
a36,a4
a37,a5
a37,a6
a38,a16
a38,a19
a39,a41
a4,a34
a4,a5
a40,a39
a41,a44
a42,a25
a44,a24
a44,a54
a45,a28
a45,a29
a46,a7
a46,a8
a47,a30
a47,a31
a48,a10
a48,a9
a49,a32
a49,a43
a5,a36
a5,a6
a50,a11
a50,a12
a51,a22
a52,a14
a53,a20
a54,a1
a54,a44
a55,a17
a55,a18
a6,a45
a6,a7
a7,a45
a7,a8
a8,a47
a8,a9
a9,a10
a9,a47
b1,b2
b1,b34
b10,b11
b10,b9
b11,b12
b11,b15
b12,b15
b12,b16
b13,b12
b13,b16
b14,b13
b15,b14
b15,b30
b16,b17
b16,b24
b17,b18
b18,b8
b18,b9
b19,b18
b19,b20
b2,b3
b2,b5
b20,b21
b20,b24
b21,b22
b21,b25
b22,b23
b22,b25
b23,b19
b23,b20
b24,b17
b24,b21
b25,b26
b25,b38
b26,b27
b26,b38
b27,b28
b27,b42
b28,b13
b29,b27
b3,b4
b3,b5
b30,b31
b31,b29
b31,b32
b32,b33
b32,b42
b33,b34
b33,b41
b34,b2
b34,b40
b35,b1
b35,b36
b36,b37
b36,b40
b37,b39
b37,b41
b38,b37
b39,b22
b39,b23
b4,b35
b40,b33
b40,b35
b41,b32
b41,b36
b42,b26
b42,b31
b5,b10
b5,b6
b6,b10
b6,b7
b7,b3
b7,b4
b8,b6
b8,b7
b9,b11
b9,b8
c1,c2
c1,c23
c10,c11
c10,c25
c11,c12
c11,c27
c12,c13
c12,c28
c13,c14
c13,c28
c14,c1
c14,c24
c15,c8
c16,c10
c16,c11
c17,c7
c18,c6
c19,c13
c19,c14
c2,c23
c2,c3
c20,c2
c20,c3
c21,c4
c21,c5
c24,c1
c26,c9
c27,c12
c3,c22
c3,c4
c4,c22
c4,c5
c5,c18
c5,c6
c6,c17
c6,c7
c7,c15
c7,c8
c8,c26
c8,c9
c9,c10
c9,c25
d1,d2
d1,d30
d10,d11
d10,d39
d11,d12
d11,d40
d12,d13
d12,d41
d13,d14
d13,d42
d14,d15
d14,d43
d15,d16
d15,d44
d16,d17
d16,d45
d17,d18
d17,d46
d18,d19
d18,d47
d19,d20
d19,d48
d2,d3
d2,d31
d20,d21
d20,d49
d21,d22
d21,d50
d22,d23
d22,d51
d23,d24
d23,d52
d24,d25
d24,d53
d25,d26
d25,d54
d26,d27
d26,d55
d27,d28
d27,d56
d28,d29
d28,d57
d3,d32
d3,d4
d30,d2
d31,d3
d32,d4
d33,d5
d34,d6
d35,d7
d36,d8
d37,d9
d38,d10
d39,d11
d4,d33
d4,d5
d40,d12
d41,d13
d42,d14
d43,d15
d44,d16
d45,d17
d46,d18
d47,d19
d48,d20
d49,d21
d5,d34
d5,d6
d50,d22
d51,d23
d52,d24
d53,d25
d54,d26
d55,d27
d56,d28
d57,d29
d6,d35
d6,d7
d7,d36
d7,d8
d8,d37
d8,d9
d9,d10
d9,d38
e1,e2
e1,e5
e10,e95
e10,e96
e11,e13
e11,e3
e12,e11
e12,e14
e13,e10
e13,e12
e14,e15
e14,e40
e15,e16
e15,e40
e16,e17
e16,e28
e17,e12
e17,e14
e18,e19
e18,e20
e19,e20
e19,e53
e2,e18
e2,e3
e20,e22
e21,e1
e22,e21
e22,e26
e23,e22
e23,e27
e24,e23
e24,e25
e25,e55
e25,e85
e26,e23
e26,e94
e27,e24
e27,e51
e28,e17
e28,e52
e29,e15
e29,e16
e3,e18
e3,e96
e30,e29
e30,e86
e31,e30
e31,e86
e32,e31
e32,e84
e33,e32
e33,e84
e34,e33
e34,e78
e35,e34
e36,e35
e36,e76
e37,e36
e37,e76
e38,e37
e38,e75
e39,e38
e39,e41
e4,e1
e4,e5
e40,e39
e40,e41
e41,e42
e41,e47
e42,e43
e43,e44
e43,e87
e44,e45
e44,e49
e45,e46
e45,e48
e46,e43
e46,e47
e47,e79
e47,e91
e48,e46
e48,e92
e49,e45
e49,e69
e5,e6
e5,e93
e50,e71
e51,e54
e52,e29
e52,e30
e53,e28
e53,e52
e54,e26
e54,e94
e55,e56
e55,e85
e56,e57
e56,e78
e57,e58
e58,e59
e58,e77
e59,e60
e59,e77
e6,e7
e6,e93
e60,e61
e60,e75
e61,e62
e61,e72
e62,e63
e62,e72
e63,e64
e63,e65
e64,e24
e64,e25
e65,e64
e65,e73
e66,e65
e66,e67
e67,e66
e67,e70
e68,e67
e68,e81
e69,e68
e69,e81
e7,e8
e7,e87
e70,e50
e70,e68
e71,e73
e71,e74
e72,e50
e72,e70
e73,e66
e73,e74
e74,e62
e74,e63
e75,e37
e75,e61
e76,e59
e76,e60
e77,e35
e77,e36
e78,e33
e78,e57
e79,e38
e79,e39
e8,e89
e8,e9
e80,e79
e80,e91
e81,e80
e81,e82
e82,e80
e82,e83
e83,e49
e83,e69
e84,e55
e84,e56
e85,e31
e85,e32
e86,e51
e86,e54
e87,e44
e87,e8
e88,e89
e88,e90
e89,e9
e89,e90
e9,e10
e9,e13
e90,e6
e90,e7
e91,e48
e92,e82
e92,e83
e93,e88
e93,e95
e94,e19
e94,e53
e95,e4
e95,e88
e96,e11
e96,e4
@.
*opposite_r(Elt,Elt)
a11,a3
a9,a3
a31,a25
a13,a1
a15,a1
a17,a1
a19,a1
a22,a1
a23,a1
a32,a22
a33,a23
a34,a54
a37,a45
a39,a42
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b13,b27
b15,b31
b10,b33
b8,b36
b5,b34
b7,b35
b18,b37
b16,b26
b11,b32
b19,b39
b24,b25
c6,c12
c2,c14
c3,c5
c8,c10
c10,c14
c11,c7
c13,c5
c15,c16
c27,c17
c28,c18
c20,c19
c21,c22
c23,c24
c25,c26
d5,d7
d9,d11
d13,d15
d17,d19
d21,d23
d25,d27
e4,e2
e96,e2
e8,e43
e13,e43
e41,e14
e83,e45
e90,e93
e81,e70
e63,e25
e62,e60
e57,e55
e32,e34
@.
*equal_r(Elt,Elt)
a16,a18
a29,a7
a31,a9
a33,a23
a34,a54
a36,a37
a37,a45
a38,a55
a42,a43
a45,a46
a46,a47
a47,a48
a48,a49
a49,a50
a50,a51
a51,a52
a52,a53
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b25,b39
b38,b22
b20,b17
b3,b6
b9,b12
b29,b14
b40,b1
b42,b41
c6,c12
c2,c14
c10,c14
c15,c17
c17,c18
c18,c21
c21,c22
c22,c20
c20,c23
c23,c24
c19,c20
c28,c19
c27,c28
c16,c27
c25,c16
c26,c25
d5,d7
d9,d11
d17,d19
d21,d23
d30,d31
d31,d32
d32,d33
d34,d35
d36,d37
d38,d39
d40,d41
d42,d43
d44,d45
d46,d47
d48,d49
d50,d51
d52,d53
d54,d55
d55,d56
d56,d57
e90,e93
e93,e4
e13,e96
e11,e10
e41,e79
e39,e47
e75,e76
e76,e77
e77,e78
e78,e84
e84,e85
e14,e16
e2,e20
e18,e21
e60,e37
e59,e36
e35,e58
e57,e34
e33,e56
@.
mesh(Elt, No) #-
b1,6
b2,1
b3,6
b4,1
b5,1
b6,6
b7,1
b8,2
b9,6
b10,2
b11,6
b12,6
b13,3
b14,6
b15,3
b16,3
b17,8
b18,3
b19,7
b20,8
b21,1
b22,8
b23,1
b24,7
b25,7
b26,2
b27,2
b28,1
b29,6
b30,1
b31,2
b32,4
b33,2
b34,2
b35,2
b36,2
b37,1
b38,8
b39,7
b40,6
b41,6
b42,6
c1,1
c2,2
c3,1
c4,1
c5,3
c6,2
c7,2
c8,3
c9,1
c10,2
c11,1
c12,2
c13,1
c14,2
c15,8
c16,8
c17,8
c18,8
c19,8
c20,8
c21,8
c22,8
c23,8
c24,8
c25,8
c26,8
c27,8
c28,8
d1,2
d2,4
d3,1
d4,1
d5,1
d6,2
d7,1
d8,2
d9,1
d10,2
d11,1
d12,2
d13,1
d14,2
d15,2
d16,2
d17,1
d18,2
d19,1
d20,2
d21,1
d22,2
d23,1
d24,1
d25,1
d26,1
d27,2
d28,4
d29,2
d30,12
d31,12
d32,12
d33,12
d34,12
d35,12
d36,12
d37,12
d38,12
d39,12
d40,12
d41,12
d42,12
d43,12
d44,12
d45,12
d46,12
d47,12
d48,12
d49,12
d50,12
d51,12
d52,12
d53,12
d54,12
d55,12
d56,12
d57,12
e1,1
e2,3
e3,1
e4,2
e5,2
e6,3
e7,2
e8,2
e9,1
e10,4
e11,4
e12,1
e13,1
e14,5
e15,2
e16,5
e17,2
e18,3
e19,10
e20,3
e21,3
e22,12
e23,2
e24,4
e25,2
e26,3
e27,3
e28,5
e29,2
e30,1
e31,1
e32,2
e33,1
e34,2
e35,1
e36,1
e37,2
e38,1
e39,6
e40,2
e41,6
e42,1
e43,5
e44,2
e45,5
e46,2
e47,6
e48,2
e49,5
e50,2
e51,1
e52,3
e53,1
e54,1
e55,2
e56,1
e57,2
e58,1
e59,1
e60,2
e61,1
e62,2
e63,3
e64,1
e65,4
e66,2
e67,2
e68,1
e69,2
e70,2
e71,2
e72,3
e73,2
e74,4
e75,9
e76,9
e77,9
e78,9
e79,6
e80,5
e81,2
e82,2
e83,2
e84,9
e85,9
e86,5
e87,1
e88,3
e89,2
e90,2
e91,2
e92,5
e93,2
e94,2
e95,2
e96,1
@.

mesh
a2,1
a3,8
a4,1
a5,1
a6,2
a7,1
a8,1
a9,3
a10,1
a11,3
a12,1
a13,1
a14,1
a15,4
a16,1
a17,2
a18,1
a19,4
a20,1
a21,1
a22,2
a23,2
a24,1
a25,2
a26,1
a27,1
a28,2
a29,1
a30,1
a31,3
a32,2
a33,2
a34,11
a35,1
a36,12
a37,12
a38,12
a39,5
a40,2
a41,1
a42,5
a43,5
a44,1
a45,12
a46,12
a47,12
a48,12
a49,12
a50,12
a51,12
a52,12
a53,12
a54,11
a55,12
@.
@//E*O*F Release2/Examples/mesh_a.d//
chmod u=r,g=r,o=r Release2/Examples/mesh_a.d
 
echo x - Release2/Examples/mesh_b.d
sed 's/^@//' > "Release2/Examples/mesh_b.d" <<'@//E*O*F Release2/Examples/mesh_b.d//'
#Elt:
 a1, a2, a3, a4, a5, a6, a7, a8, a9,a10,
a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,
a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,
a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,
a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,
a51,a52,a53,a54,a55,
 b1, b2, b3, b4, b5, b6, b7, b8, b9,b10,
b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,
b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,
b41,b42,
 c1, c2, c3, c4, c5, c6, c7, c8, c9,c10,
c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
c21,c22,c23,c24,c25,c26,c27,c28,
 d1, d2, d3, d4, d5, d6, d7, d8, d9,d10,
d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,
d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,
d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,
d51,d52,d53,d54,d55,d56,d57,
 e1, e2, e3, e4, e5, e6, e7, e8, e9,e10,
e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,
e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,
e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,
e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,
e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,
e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,
e81,e82,e83,e84,e85,e86,e87,e88,e89,e90,
e91,e92,e93,e94,e95,e96.
No:
*0,*1,*2,*3,*4,*5,*6,*7,*8,*9,*10,*11,*12.

*long(Elt)
a1
a34
a54
b19
b39
e19
e22
@.
*usual(Elt)
a3
a39
b11
b13
b15
b24
b25
b27
b31
b32
c5
c6
c8
c10
c12
c14
d1
d2
d28
d29
e2
e4
e5
e6
e7
e8
e14
e15
e16
e17
e18
e20
e21
e24
e25
e26
e27
e28
e29
e32
e34
e35
e37
e40
e43
e44
e45
e46
e48
e49
e50
e52
e55
e57
e58
e60
e62
e63
e65
e66
e67
e69
e70
e71
e72
e73
e74
e80
e81
e83
e86
e88
e89
e90
e91
e92
e93
e94
e95
@.
*short(Elt)
a6
a9
a11
a13
a15
a19
a22
a25
a26
a28
a31
a35
a40
a44
b5
b7
b8
b10
b16
b18
b26
b33
b36
b37
c7
c13
d3
d4
d6
d8
d10
d12
d14
d15
d16
d18
d20
d22
d24
d26
d27
e1
e3
e9
e12
e23
e30
e31
e33
e36
e38
e51
e53
e54
e56
e59
e61
e64
e68
e82
e87
@.
*circuit(Elt)
c15
c16
c17
c18
c19
c24
c25
c26
c27
c28
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
@.
*half_circuit(Elt)
a36
a37
a45
a46
a47
a48
a49
a50
a51
a52
a53
b3
b6
b9
b12
b17
b20
b41
b42
@.
*quarter_circuit(Elt)
e75
e76
e77
e78
e84
e85
@.
*short_for_hole(Elt)
a16
a17
a18
a23
a33
b28
b30
b34
b35
c3
c4
e13
e96
@.
*long_for_hole(Elt)
c2
e41
e79
@.
*circuit_hole(Elt)
c20
c21
c22
c23
@.
*half_circuit_hole(Elt)
a38
a42
a43
a55
b1
b14
b22
b29
b38
b40
e10
e11
e39
e47
@.
*not_important(Elt)
a2
a4
a5
a7
a8
a10
a12
a14
a20
a21
a24
a27
a29
a30
a32
a41
b2
b4
b21
b23
c1
c9
c11
d5
d7
d9
d11
d13
d17
d19
d21
d23
d25
e42
@.
*free(Elt)
a39
a40
c6
c7
c11
c12
c13
c15
c16
c17
c18
c19
c20
c27
c28
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d17
d18
d19
d20
d21
d22
d23
d24
d25
d26
d27
d29
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
e6
e16
e19
e20
e22
e23
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e36
e45
e48
e49
e51
e52
e53
e54
e66
e71
e73
e83
e86
e88
e90
e92
e93
e94
@.
*one_side_fixed(Elt)
a34
a35
a41
c2
c3
c5
c8
c10
c14
e5
e7
e15
e17
e18
e21
e24
e37
e44
e46
e50
e65
e67
e69
e74
e76
e77
e78
e82
e84
e85
e89
e91
e95
@.
*two_side_fixed(Elt)
a36
a37
a38
a42
a43
a45
a46
a47
a48
a49
a50
a51
a52
a53
a55
b6
b9
b12
b14
b17
b20
b22
b29
b38
b40
b41
b42
e10
e11
e39
e47
@.
*fixed(Elt)
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24
a25
a26
a27
a28
a29
a30
a31
a32
a33
a44
a54
b1
b2
b3
b4
b5
b7
b8
b10
b11
b13
b15
b16
b18
b19
b21
b23
b24
b25
b26
b27
b28
b30
b31
b32
b33
b34
b35
b36
b37
b39
c1
c4
c9
c21
c22
c23
c24
c25
c26
d1
d2
d28
d30
e1
e2
e3
e4
e8
e9
e12
e13
e14
e25
e38
e40
e41
e42
e43
e55
e56
e57
e58
e59
e60
e61
e62
e63
e64
e68
e70
e72
e75
e79
e80
e81
e87
e96
@.
*not_loaded(Elt)
a1
a2
a3
a4
a5
a6
a7
a23
a24
a25
a26
a27
a28
a29
a33
a36
a37
a42
a44
a45
a46
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12
b13
b14
b15
b16
b17
b18
b19
b20
b21
b23
b24
c1
c2
c3
c4
c5
c6
c7
c8
c9
c15
c17
c18
c20
c21
c22
c23
c26
d1
d2
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d20
d21
d22
d23
d24
d25
d26
d27
d28
d29
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d49
d50
d51
d52
d53
d54
d55
d56
d57
e1
e2
e3
e4
e5
e6
e7
e8
e9
e10
e11
e12
e13
e14
e15
e16
e17
e18
e19
e20
e21
e22
e23
e24
e25
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e39
e40
e41
e42
e43
e44
e45
e46
e47
e48
e49
e50
e51
e52
e53
e54
e55
e56
e57
e58
e62
e63
e64
e65
e66
e67
e68
e69
e70
e71
e72
e73
e74
e77
e78
e79
e80
e81
e82
e83
e84
e85
e86
e87
e88
e89
e90
e91
e92
e93
e94
e95
e96
@.
*one_side_loaded(Elt)
a34
a35
a40
a41
a54
d45
d46
d47
d48
e36
e38
e59
e61
@.
*two_side_loaded(Elt)
e37
@.
*cont_loaded(Elt)
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a30
a31
a32
a38
a39
a43
a47
a48
a49
a50
a51
a52
a53
b22
b25
b26
b27
b28
b29
b30
b31
b32
b33
b34
b35
b36
b37
b38
b39
b40
b41
b42
c10
c11
c12
c13
c14
c16
c19
c24
c25
c27
c28
d17
d18
d19
e60
e75
e76
@.
*neighbour(Elt,Elt)
a1,a2
a1,a44
a10,a11
a10,a49
a11,a12
a11,a49
a12,a13
a12,a51
a13,a14
a14,a15
a14,a53
a15,a16
a15,a38
a16,a17
a16,a55
a17,a18
a18,a19
a18,a38
a19,a20
a2,a3
a2,a54
a20,a21
a20,a52
a21,a22
a22,a23
a22,a50
a23,a24
a24,a1
a24,a42
a25,a26
a25,a35
a26,a27
a26,a36
a27,a28
a27,a37
a28,a29
a28,a37
a29,a30
a29,a46
a3,a4
a30,a31
a30,a46
a31,a32
a31,a48
a32,a33
a32,a48
a33,a25
a34,a35
a34,a40
a35,a26
a35,a40
a36,a27
a36,a4
a37,a5
a37,a6
a38,a16
a38,a19
a39,a41
a4,a34
a4,a5
a40,a39
a41,a44
a42,a25
a44,a24
a44,a54
a45,a28
a45,a29
a46,a7
a46,a8
a47,a30
a47,a31
a48,a10
a48,a9
a49,a32
a49,a43
a5,a36
a5,a6
a50,a11
a50,a12
a51,a22
a52,a14
a53,a20
a54,a1
a54,a44
a55,a17
a55,a18
a6,a45
a6,a7
a7,a45
a7,a8
a8,a47
a8,a9
a9,a10
a9,a47
b1,b2
b1,b34
b10,b11
b10,b9
b11,b12
b11,b15
b12,b15
b12,b16
b13,b12
b13,b16
b14,b13
b15,b14
b15,b30
b16,b17
b16,b24
b17,b18
b18,b8
b18,b9
b19,b18
b19,b20
b2,b3
b2,b5
b20,b21
b20,b24
b21,b22
b21,b25
b22,b23
b22,b25
b23,b19
b23,b20
b24,b17
b24,b21
b25,b26
b25,b38
b26,b27
b26,b38
b27,b28
b27,b42
b28,b13
b29,b27
b3,b4
b3,b5
b30,b31
b31,b29
b31,b32
b32,b33
b32,b42
b33,b34
b33,b41
b34,b2
b34,b40
b35,b1
b35,b36
b36,b37
b36,b40
b37,b39
b37,b41
b38,b37
b39,b22
b39,b23
b4,b35
b40,b33
b40,b35
b41,b32
b41,b36
b42,b26
b42,b31
b5,b10
b5,b6
b6,b10
b6,b7
b7,b3
b7,b4
b8,b6
b8,b7
b9,b11
b9,b8
c1,c2
c1,c23
c10,c11
c10,c25
c11,c12
c11,c27
c12,c13
c12,c28
c13,c14
c13,c28
c14,c1
c14,c24
c15,c8
c16,c10
c16,c11
c17,c7
c18,c6
c19,c13
c19,c14
c2,c23
c2,c3
c20,c2
c20,c3
c21,c4
c21,c5
c24,c1
c26,c9
c27,c12
c3,c22
c3,c4
c4,c22
c4,c5
c5,c18
c5,c6
c6,c17
c6,c7
c7,c15
c7,c8
c8,c26
c8,c9
c9,c10
c9,c25
d1,d2
d1,d30
d10,d11
d10,d39
d11,d12
d11,d40
d12,d13
d12,d41
d13,d14
d13,d42
d14,d15
d14,d43
d15,d16
d15,d44
d16,d17
d16,d45
d17,d18
d17,d46
d18,d19
d18,d47
d19,d20
d19,d48
d2,d3
d2,d31
d20,d21
d20,d49
d21,d22
d21,d50
d22,d23
d22,d51
d23,d24
d23,d52
d24,d25
d24,d53
d25,d26
d25,d54
d26,d27
d26,d55
d27,d28
d27,d56
d28,d29
d28,d57
d3,d32
d3,d4
d30,d2
d31,d3
d32,d4
d33,d5
d34,d6
d35,d7
d36,d8
d37,d9
d38,d10
d39,d11
d4,d33
d4,d5
d40,d12
d41,d13
d42,d14
d43,d15
d44,d16
d45,d17
d46,d18
d47,d19
d48,d20
d49,d21
d5,d34
d5,d6
d50,d22
d51,d23
d52,d24
d53,d25
d54,d26
d55,d27
d56,d28
d57,d29
d6,d35
d6,d7
d7,d36
d7,d8
d8,d37
d8,d9
d9,d10
d9,d38
e1,e2
e1,e5
e10,e95
e10,e96
e11,e13
e11,e3
e12,e11
e12,e14
e13,e10
e13,e12
e14,e15
e14,e40
e15,e16
e15,e40
e16,e17
e16,e28
e17,e12
e17,e14
e18,e19
e18,e20
e19,e20
e19,e53
e2,e18
e2,e3
e20,e22
e21,e1
e22,e21
e22,e26
e23,e22
e23,e27
e24,e23
e24,e25
e25,e55
e25,e85
e26,e23
e26,e94
e27,e24
e27,e51
e28,e17
e28,e52
e29,e15
e29,e16
e3,e18
e3,e96
e30,e29
e30,e86
e31,e30
e31,e86
e32,e31
e32,e84
e33,e32
e33,e84
e34,e33
e34,e78
e35,e34
e36,e35
e36,e76
e37,e36
e37,e76
e38,e37
e38,e75
e39,e38
e39,e41
e4,e1
e4,e5
e40,e39
e40,e41
e41,e42
e41,e47
e42,e43
e43,e44
e43,e87
e44,e45
e44,e49
e45,e46
e45,e48
e46,e43
e46,e47
e47,e79
e47,e91
e48,e46
e48,e92
e49,e45
e49,e69
e5,e6
e5,e93
e50,e71
e51,e54
e52,e29
e52,e30
e53,e28
e53,e52
e54,e26
e54,e94
e55,e56
e55,e85
e56,e57
e56,e78
e57,e58
e58,e59
e58,e77
e59,e60
e59,e77
e6,e7
e6,e93
e60,e61
e60,e75
e61,e62
e61,e72
e62,e63
e62,e72
e63,e64
e63,e65
e64,e24
e64,e25
e65,e64
e65,e73
e66,e65
e66,e67
e67,e66
e67,e70
e68,e67
e68,e81
e69,e68
e69,e81
e7,e8
e7,e87
e70,e50
e70,e68
e71,e73
e71,e74
e72,e50
e72,e70
e73,e66
e73,e74
e74,e62
e74,e63
e75,e37
e75,e61
e76,e59
e76,e60
e77,e35
e77,e36
e78,e33
e78,e57
e79,e38
e79,e39
e8,e89
e8,e9
e80,e79
e80,e91
e81,e80
e81,e82
e82,e80
e82,e83
e83,e49
e83,e69
e84,e55
e84,e56
e85,e31
e85,e32
e86,e51
e86,e54
e87,e44
e87,e8
e88,e89
e88,e90
e89,e9
e89,e90
e9,e10
e9,e13
e90,e6
e90,e7
e91,e48
e92,e82
e92,e83
e93,e88
e93,e95
e94,e19
e94,e53
e95,e4
e95,e88
e96,e11
e96,e4
@.
*opposite_r(Elt,Elt)
a11,a3
a9,a3
a31,a25
a13,a1
a15,a1
a17,a1
a19,a1
a22,a1
a23,a1
a32,a22
a33,a23
a34,a54
a37,a45
a39,a42
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b13,b27
b15,b31
b10,b33
b8,b36
b5,b34
b7,b35
b18,b37
b16,b26
b11,b32
b19,b39
b24,b25
c6,c12
c2,c14
c3,c5
c8,c10
c10,c14
c11,c7
c13,c5
c15,c16
c27,c17
c28,c18
c20,c19
c21,c22
c23,c24
c25,c26
d5,d7
d9,d11
d13,d15
d17,d19
d21,d23
d25,d27
e4,e2
e96,e2
e8,e43
e13,e43
e41,e14
e83,e45
e90,e93
e81,e70
e63,e25
e62,e60
e57,e55
e32,e34
@.
*equal_r(Elt,Elt)
a16,a18
a29,a7
a31,a9
a33,a23
a34,a54
a36,a37
a37,a45
a38,a55
a42,a43
a45,a46
a46,a47
a47,a48
a48,a49
a49,a50
a50,a51
a51,a52
a52,a53
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b25,b39
b38,b22
b20,b17
b3,b6
b9,b12
b29,b14
b40,b1
b42,b41
c6,c12
c2,c14
c10,c14
c15,c17
c17,c18
c18,c21
c21,c22
c22,c20
c20,c23
c23,c24
c19,c20
c28,c19
c27,c28
c16,c27
c25,c16
c26,c25
d5,d7
d9,d11
d17,d19
d21,d23
d30,d31
d31,d32
d32,d33
d34,d35
d36,d37
d38,d39
d40,d41
d42,d43
d44,d45
d46,d47
d48,d49
d50,d51
d52,d53
d54,d55
d55,d56
d56,d57
e90,e93
e93,e4
e13,e96
e11,e10
e41,e79
e39,e47
e75,e76
e76,e77
e77,e78
e78,e84
e84,e85
e14,e16
e2,e20
e18,e21
e60,e37
e59,e36
e35,e58
e57,e34
e33,e56
@.
mesh(Elt, No) #-
a2,1
a3,8
a4,1
a5,1
a6,2
a7,1
a8,1
a9,3
a10,1
a11,3
a12,1
a13,1
a14,1
a15,4
a16,1
a17,2
a18,1
a19,4
a20,1
a21,1
a22,2
a23,2
a24,1
a25,2
a26,1
a27,1
a28,2
a29,1
a30,1
a31,3
a32,2
a33,2
a34,11
a35,1
a36,12
a37,12
a38,12
a39,5
a40,2
a41,1
a42,5
a43,5
a44,1
a45,12
a46,12
a47,12
a48,12
a49,12
a50,12
a51,12
a52,12
a53,12
a54,11
a55,12
c1,1
c2,2
c3,1
c4,1
c5,3
c6,2
c7,2
c8,3
c9,1
c10,2
c11,1
c12,2
c13,1
c14,2
c15,8
c16,8
c17,8
c18,8
c19,8
c20,8
c21,8
c22,8
c23,8
c24,8
c25,8
c26,8
c27,8
c28,8
d1,2
d2,4
d3,1
d4,1
d5,1
d6,2
d7,1
d8,2
d9,1
d10,2
d11,1
d12,2
d13,1
d14,2
d15,2
d16,2
d17,1
d18,2
d19,1
d20,2
d21,1
d22,2
d23,1
d24,1
d25,1
d26,1
d27,2
d28,4
d29,2
d30,12
d31,12
d32,12
d33,12
d34,12
d35,12
d36,12
d37,12
d38,12
d39,12
d40,12
d41,12
d42,12
d43,12
d44,12
d45,12
d46,12
d47,12
d48,12
d49,12
d50,12
d51,12
d52,12
d53,12
d54,12
d55,12
d56,12
d57,12
e1,1
e2,3
e3,1
e4,2
e5,2
e6,3
e7,2
e8,2
e9,1
e10,4
e11,4
e12,1
e13,1
e14,5
e15,2
e16,5
e17,2
e18,3
e19,10
e20,3
e21,3
e22,12
e23,2
e24,4
e25,2
e26,3
e27,3
e28,5
e29,2
e30,1
e31,1
e32,2
e33,1
e34,2
e35,1
e36,1
e37,2
e38,1
e39,6
e40,2
e41,6
e42,1
e43,5
e44,2
e45,5
e46,2
e47,6
e48,2
e49,5
e50,2
e51,1
e52,3
e53,1
e54,1
e55,2
e56,1
e57,2
e58,1
e59,1
e60,2
e61,1
e62,2
e63,3
e64,1
e65,4
e66,2
e67,2
e68,1
e69,2
e70,2
e71,2
e72,3
e73,2
e74,4
e75,9
e76,9
e77,9
e78,9
e79,6
e80,5
e81,2
e82,2
e83,2
e84,9
e85,9
e86,5
e87,1
e88,3
e89,2
e90,2
e91,2
e92,5
e93,2
e94,2
e95,2
e96,1
@.

mesh
b1,6
b2,1
b3,6
b4,1
b5,1
b6,6
b7,1
b8,2
b9,6
b10,2
b11,6
b12,6
b13,3
b14,6
b15,3
b16,3
b17,8
b18,3
b19,7
b20,8
b21,1
b22,8
b23,1
b24,7
b25,7
b26,2
b27,2
b28,1
b29,6
b30,1
b31,2
b32,4
b33,2
b34,2
b35,2
b36,2
b37,1
b38,8
b39,7
b40,6
b41,6
b42,6
@.
@//E*O*F Release2/Examples/mesh_b.d//
chmod u=r,g=r,o=r Release2/Examples/mesh_b.d
 
echo x - Release2/Examples/mesh_c.d
sed 's/^@//' > "Release2/Examples/mesh_c.d" <<'@//E*O*F Release2/Examples/mesh_c.d//'
#Elt:
 a1, a2, a3, a4, a5, a6, a7, a8, a9,a10,
a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,
a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,
a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,
a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,
a51,a52,a53,a54,a55,
 b1, b2, b3, b4, b5, b6, b7, b8, b9,b10,
b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,
b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,
b41,b42,
 c1, c2, c3, c4, c5, c6, c7, c8, c9,c10,
c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
c21,c22,c23,c24,c25,c26,c27,c28,
 d1, d2, d3, d4, d5, d6, d7, d8, d9,d10,
d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,
d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,
d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,
d51,d52,d53,d54,d55,d56,d57,
 e1, e2, e3, e4, e5, e6, e7, e8, e9,e10,
e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,
e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,
e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,
e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,
e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,
e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,
e81,e82,e83,e84,e85,e86,e87,e88,e89,e90,
e91,e92,e93,e94,e95,e96.
No:
*0,*1,*2,*3,*4,*5,*6,*7,*8,*9,*10,*11,*12.

*long(Elt)
a1
a34
a54
b19
b39
e19
e22
@.
*usual(Elt)
a3
a39
b11
b13
b15
b24
b25
b27
b31
b32
c5
c6
c8
c10
c12
c14
d1
d2
d28
d29
e2
e4
e5
e6
e7
e8
e14
e15
e16
e17
e18
e20
e21
e24
e25
e26
e27
e28
e29
e32
e34
e35
e37
e40
e43
e44
e45
e46
e48
e49
e50
e52
e55
e57
e58
e60
e62
e63
e65
e66
e67
e69
e70
e71
e72
e73
e74
e80
e81
e83
e86
e88
e89
e90
e91
e92
e93
e94
e95
@.
*short(Elt)
a6
a9
a11
a13
a15
a19
a22
a25
a26
a28
a31
a35
a40
a44
b5
b7
b8
b10
b16
b18
b26
b33
b36
b37
c7
c13
d3
d4
d6
d8
d10
d12
d14
d15
d16
d18
d20
d22
d24
d26
d27
e1
e3
e9
e12
e23
e30
e31
e33
e36
e38
e51
e53
e54
e56
e59
e61
e64
e68
e82
e87
@.
*circuit(Elt)
c15
c16
c17
c18
c19
c24
c25
c26
c27
c28
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
@.
*half_circuit(Elt)
a36
a37
a45
a46
a47
a48
a49
a50
a51
a52
a53
b3
b6
b9
b12
b17
b20
b41
b42
@.
*quarter_circuit(Elt)
e75
e76
e77
e78
e84
e85
@.
*short_for_hole(Elt)
a16
a17
a18
a23
a33
b28
b30
b34
b35
c3
c4
e13
e96
@.
*long_for_hole(Elt)
c2
e41
e79
@.
*circuit_hole(Elt)
c20
c21
c22
c23
@.
*half_circuit_hole(Elt)
a38
a42
a43
a55
b1
b14
b22
b29
b38
b40
e10
e11
e39
e47
@.
*not_important(Elt)
a2
a4
a5
a7
a8
a10
a12
a14
a20
a21
a24
a27
a29
a30
a32
a41
b2
b4
b21
b23
c1
c9
c11
d5
d7
d9
d11
d13
d17
d19
d21
d23
d25
e42
@.
*free(Elt)
a39
a40
c6
c7
c11
c12
c13
c15
c16
c17
c18
c19
c20
c27
c28
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d17
d18
d19
d20
d21
d22
d23
d24
d25
d26
d27
d29
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
e6
e16
e19
e20
e22
e23
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e36
e45
e48
e49
e51
e52
e53
e54
e66
e71
e73
e83
e86
e88
e90
e92
e93
e94
@.
*one_side_fixed(Elt)
a34
a35
a41
c2
c3
c5
c8
c10
c14
e5
e7
e15
e17
e18
e21
e24
e37
e44
e46
e50
e65
e67
e69
e74
e76
e77
e78
e82
e84
e85
e89
e91
e95
@.
*two_side_fixed(Elt)
a36
a37
a38
a42
a43
a45
a46
a47
a48
a49
a50
a51
a52
a53
a55
b6
b9
b12
b14
b17
b20
b22
b29
b38
b40
b41
b42
e10
e11
e39
e47
@.
*fixed(Elt)
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24
a25
a26
a27
a28
a29
a30
a31
a32
a33
a44
a54
b1
b2
b3
b4
b5
b7
b8
b10
b11
b13
b15
b16
b18
b19
b21
b23
b24
b25
b26
b27
b28
b30
b31
b32
b33
b34
b35
b36
b37
b39
c1
c4
c9
c21
c22
c23
c24
c25
c26
d1
d2
d28
d30
e1
e2
e3
e4
e8
e9
e12
e13
e14
e25
e38
e40
e41
e42
e43
e55
e56
e57
e58
e59
e60
e61
e62
e63
e64
e68
e70
e72
e75
e79
e80
e81
e87
e96
@.
*not_loaded(Elt)
a1
a2
a3
a4
a5
a6
a7
a23
a24
a25
a26
a27
a28
a29
a33
a36
a37
a42
a44
a45
a46
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12
b13
b14
b15
b16
b17
b18
b19
b20
b21
b23
b24
c1
c2
c3
c4
c5
c6
c7
c8
c9
c15
c17
c18
c20
c21
c22
c23
c26
d1
d2
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d20
d21
d22
d23
d24
d25
d26
d27
d28
d29
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d49
d50
d51
d52
d53
d54
d55
d56
d57
e1
e2
e3
e4
e5
e6
e7
e8
e9
e10
e11
e12
e13
e14
e15
e16
e17
e18
e19
e20
e21
e22
e23
e24
e25
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e39
e40
e41
e42
e43
e44
e45
e46
e47
e48
e49
e50
e51
e52
e53
e54
e55
e56
e57
e58
e62
e63
e64
e65
e66
e67
e68
e69
e70
e71
e72
e73
e74
e77
e78
e79
e80
e81
e82
e83
e84
e85
e86
e87
e88
e89
e90
e91
e92
e93
e94
e95
e96
@.
*one_side_loaded(Elt)
a34
a35
a40
a41
a54
d45
d46
d47
d48
e36
e38
e59
e61
@.
*two_side_loaded(Elt)
e37
@.
*cont_loaded(Elt)
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a30
a31
a32
a38
a39
a43
a47
a48
a49
a50
a51
a52
a53
b22
b25
b26
b27
b28
b29
b30
b31
b32
b33
b34
b35
b36
b37
b38
b39
b40
b41
b42
c10
c11
c12
c13
c14
c16
c19
c24
c25
c27
c28
d17
d18
d19
e60
e75
e76
@.
*neighbour(Elt,Elt)
a1,a2
a1,a44
a10,a11
a10,a49
a11,a12
a11,a49
a12,a13
a12,a51
a13,a14
a14,a15
a14,a53
a15,a16
a15,a38
a16,a17
a16,a55
a17,a18
a18,a19
a18,a38
a19,a20
a2,a3
a2,a54
a20,a21
a20,a52
a21,a22
a22,a23
a22,a50
a23,a24
a24,a1
a24,a42
a25,a26
a25,a35
a26,a27
a26,a36
a27,a28
a27,a37
a28,a29
a28,a37
a29,a30
a29,a46
a3,a4
a30,a31
a30,a46
a31,a32
a31,a48
a32,a33
a32,a48
a33,a25
a34,a35
a34,a40
a35,a26
a35,a40
a36,a27
a36,a4
a37,a5
a37,a6
a38,a16
a38,a19
a39,a41
a4,a34
a4,a5
a40,a39
a41,a44
a42,a25
a44,a24
a44,a54
a45,a28
a45,a29
a46,a7
a46,a8
a47,a30
a47,a31
a48,a10
a48,a9
a49,a32
a49,a43
a5,a36
a5,a6
a50,a11
a50,a12
a51,a22
a52,a14
a53,a20
a54,a1
a54,a44
a55,a17
a55,a18
a6,a45
a6,a7
a7,a45
a7,a8
a8,a47
a8,a9
a9,a10
a9,a47
b1,b2
b1,b34
b10,b11
b10,b9
b11,b12
b11,b15
b12,b15
b12,b16
b13,b12
b13,b16
b14,b13
b15,b14
b15,b30
b16,b17
b16,b24
b17,b18
b18,b8
b18,b9
b19,b18
b19,b20
b2,b3
b2,b5
b20,b21
b20,b24
b21,b22
b21,b25
b22,b23
b22,b25
b23,b19
b23,b20
b24,b17
b24,b21
b25,b26
b25,b38
b26,b27
b26,b38
b27,b28
b27,b42
b28,b13
b29,b27
b3,b4
b3,b5
b30,b31
b31,b29
b31,b32
b32,b33
b32,b42
b33,b34
b33,b41
b34,b2
b34,b40
b35,b1
b35,b36
b36,b37
b36,b40
b37,b39
b37,b41
b38,b37
b39,b22
b39,b23
b4,b35
b40,b33
b40,b35
b41,b32
b41,b36
b42,b26
b42,b31
b5,b10
b5,b6
b6,b10
b6,b7
b7,b3
b7,b4
b8,b6
b8,b7
b9,b11
b9,b8
c1,c2
c1,c23
c10,c11
c10,c25
c11,c12
c11,c27
c12,c13
c12,c28
c13,c14
c13,c28
c14,c1
c14,c24
c15,c8
c16,c10
c16,c11
c17,c7
c18,c6
c19,c13
c19,c14
c2,c23
c2,c3
c20,c2
c20,c3
c21,c4
c21,c5
c24,c1
c26,c9
c27,c12
c3,c22
c3,c4
c4,c22
c4,c5
c5,c18
c5,c6
c6,c17
c6,c7
c7,c15
c7,c8
c8,c26
c8,c9
c9,c10
c9,c25
d1,d2
d1,d30
d10,d11
d10,d39
d11,d12
d11,d40
d12,d13
d12,d41
d13,d14
d13,d42
d14,d15
d14,d43
d15,d16
d15,d44
d16,d17
d16,d45
d17,d18
d17,d46
d18,d19
d18,d47
d19,d20
d19,d48
d2,d3
d2,d31
d20,d21
d20,d49
d21,d22
d21,d50
d22,d23
d22,d51
d23,d24
d23,d52
d24,d25
d24,d53
d25,d26
d25,d54
d26,d27
d26,d55
d27,d28
d27,d56
d28,d29
d28,d57
d3,d32
d3,d4
d30,d2
d31,d3
d32,d4
d33,d5
d34,d6
d35,d7
d36,d8
d37,d9
d38,d10
d39,d11
d4,d33
d4,d5
d40,d12
d41,d13
d42,d14
d43,d15
d44,d16
d45,d17
d46,d18
d47,d19
d48,d20
d49,d21
d5,d34
d5,d6
d50,d22
d51,d23
d52,d24
d53,d25
d54,d26
d55,d27
d56,d28
d57,d29
d6,d35
d6,d7
d7,d36
d7,d8
d8,d37
d8,d9
d9,d10
d9,d38
e1,e2
e1,e5
e10,e95
e10,e96
e11,e13
e11,e3
e12,e11
e12,e14
e13,e10
e13,e12
e14,e15
e14,e40
e15,e16
e15,e40
e16,e17
e16,e28
e17,e12
e17,e14
e18,e19
e18,e20
e19,e20
e19,e53
e2,e18
e2,e3
e20,e22
e21,e1
e22,e21
e22,e26
e23,e22
e23,e27
e24,e23
e24,e25
e25,e55
e25,e85
e26,e23
e26,e94
e27,e24
e27,e51
e28,e17
e28,e52
e29,e15
e29,e16
e3,e18
e3,e96
e30,e29
e30,e86
e31,e30
e31,e86
e32,e31
e32,e84
e33,e32
e33,e84
e34,e33
e34,e78
e35,e34
e36,e35
e36,e76
e37,e36
e37,e76
e38,e37
e38,e75
e39,e38
e39,e41
e4,e1
e4,e5
e40,e39
e40,e41
e41,e42
e41,e47
e42,e43
e43,e44
e43,e87
e44,e45
e44,e49
e45,e46
e45,e48
e46,e43
e46,e47
e47,e79
e47,e91
e48,e46
e48,e92
e49,e45
e49,e69
e5,e6
e5,e93
e50,e71
e51,e54
e52,e29
e52,e30
e53,e28
e53,e52
e54,e26
e54,e94
e55,e56
e55,e85
e56,e57
e56,e78
e57,e58
e58,e59
e58,e77
e59,e60
e59,e77
e6,e7
e6,e93
e60,e61
e60,e75
e61,e62
e61,e72
e62,e63
e62,e72
e63,e64
e63,e65
e64,e24
e64,e25
e65,e64
e65,e73
e66,e65
e66,e67
e67,e66
e67,e70
e68,e67
e68,e81
e69,e68
e69,e81
e7,e8
e7,e87
e70,e50
e70,e68
e71,e73
e71,e74
e72,e50
e72,e70
e73,e66
e73,e74
e74,e62
e74,e63
e75,e37
e75,e61
e76,e59
e76,e60
e77,e35
e77,e36
e78,e33
e78,e57
e79,e38
e79,e39
e8,e89
e8,e9
e80,e79
e80,e91
e81,e80
e81,e82
e82,e80
e82,e83
e83,e49
e83,e69
e84,e55
e84,e56
e85,e31
e85,e32
e86,e51
e86,e54
e87,e44
e87,e8
e88,e89
e88,e90
e89,e9
e89,e90
e9,e10
e9,e13
e90,e6
e90,e7
e91,e48
e92,e82
e92,e83
e93,e88
e93,e95
e94,e19
e94,e53
e95,e4
e95,e88
e96,e11
e96,e4
@.
*opposite_r(Elt,Elt)
a11,a3
a9,a3
a31,a25
a13,a1
a15,a1
a17,a1
a19,a1
a22,a1
a23,a1
a32,a22
a33,a23
a34,a54
a37,a45
a39,a42
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b13,b27
b15,b31
b10,b33
b8,b36
b5,b34
b7,b35
b18,b37
b16,b26
b11,b32
b19,b39
b24,b25
c6,c12
c2,c14
c3,c5
c8,c10
c10,c14
c11,c7
c13,c5
c15,c16
c27,c17
c28,c18
c20,c19
c21,c22
c23,c24
c25,c26
d5,d7
d9,d11
d13,d15
d17,d19
d21,d23
d25,d27
e4,e2
e96,e2
e8,e43
e13,e43
e41,e14
e83,e45
e90,e93
e81,e70
e63,e25
e62,e60
e57,e55
e32,e34
@.
*equal_r(Elt,Elt)
a16,a18
a29,a7
a31,a9
a33,a23
a34,a54
a36,a37
a37,a45
a38,a55
a42,a43
a45,a46
a46,a47
a47,a48
a48,a49
a49,a50
a50,a51
a51,a52
a52,a53
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b25,b39
b38,b22
b20,b17
b3,b6
b9,b12
b29,b14
b40,b1
b42,b41
c6,c12
c2,c14
c10,c14
c15,c17
c17,c18
c18,c21
c21,c22
c22,c20
c20,c23
c23,c24
c19,c20
c28,c19
c27,c28
c16,c27
c25,c16
c26,c25
d5,d7
d9,d11
d17,d19
d21,d23
d30,d31
d31,d32
d32,d33
d34,d35
d36,d37
d38,d39
d40,d41
d42,d43
d44,d45
d46,d47
d48,d49
d50,d51
d52,d53
d54,d55
d55,d56
d56,d57
e90,e93
e93,e4
e13,e96
e11,e10
e41,e79
e39,e47
e75,e76
e76,e77
e77,e78
e78,e84
e84,e85
e14,e16
e2,e20
e18,e21
e60,e37
e59,e36
e35,e58
e57,e34
e33,e56
@.
mesh(Elt, No)
a2,1
a3,8
a4,1
a5,1
a6,2
a7,1
a8,1
a9,3
a10,1
a11,3
a12,1
a13,1
a14,1
a15,4
a16,1
a17,2
a18,1
a19,4
a20,1
a21,1
a22,2
a23,2
a24,1
a25,2
a26,1
a27,1
a28,2
a29,1
a30,1
a31,3
a32,2
a33,2
a34,11
a35,1
a36,12
a37,12
a38,12
a39,5
a40,2
a41,1
a42,5
a43,5
a44,1
a45,12
a46,12
a47,12
a48,12
a49,12
a50,12
a51,12
a52,12
a53,12
a54,11
a55,12
b1,6
b2,1
b3,6
b4,1
b5,1
b6,6
b7,1
b8,2
b9,6
b10,2
b11,6
b12,6
b13,3
b14,6
b15,3
b16,3
b17,8
b18,3
b19,7
b20,8
b21,1
b22,8
b23,1
b24,7
b25,7
b26,2
b27,2
b28,1
b29,6
b30,1
b31,2
b32,4
b33,2
b34,2
b35,2
b36,2
b37,1
b38,8
b39,7
b40,6
b41,6
b42,6
d1,2
d2,4
d3,1
d4,1
d5,1
d6,2
d7,1
d8,2
d9,1
d10,2
d11,1
d12,2
d13,1
d14,2
d15,2
d16,2
d17,1
d18,2
d19,1
d20,2
d21,1
d22,2
d23,1
d24,1
d25,1
d26,1
d27,2
d28,4
d29,2
d30,12
d31,12
d32,12
d33,12
d34,12
d35,12
d36,12
d37,12
d38,12
d39,12
d40,12
d41,12
d42,12
d43,12
d44,12
d45,12
d46,12
d47,12
d48,12
d49,12
d50,12
d51,12
d52,12
d53,12
d54,12
d55,12
d56,12
d57,12
e1,1
e2,3
e3,1
e4,2
e5,2
e6,3
e7,2
e8,2
e9,1
e10,4
e11,4
e12,1
e13,1
e14,5
e15,2
e16,5
e17,2
e18,3
e19,10
e20,3
e21,3
e22,12
e23,2
e24,4
e25,2
e26,3
e27,3
e28,5
e29,2
e30,1
e31,1
e32,2
e33,1
e34,2
e35,1
e36,1
e37,2
e38,1
e39,6
e40,2
e41,6
e42,1
e43,5
e44,2
e45,5
e46,2
e47,6
e48,2
e49,5
e50,2
e51,1
e52,3
e53,1
e54,1
e55,2
e56,1
e57,2
e58,1
e59,1
e60,2
e61,1
e62,2
e63,3
e64,1
e65,4
e66,2
e67,2
e68,1
e69,2
e70,2
e71,2
e72,3
e73,2
e74,4
e75,9
e76,9
e77,9
e78,9
e79,6
e80,5
e81,2
e82,2
e83,2
e84,9
e85,9
e86,5
e87,1
e88,3
e89,2
e90,2
e91,2
e92,5
e93,2
e94,2
e95,2
e96,1
@.

mesh
c1,1
c2,2
c3,1
c4,1
c5,3
c6,2
c7,2
c8,3
c9,1
c10,2
c11,1
c12,2
c13,1
c14,2
c15,8
c16,8
c17,8
c18,8
c19,8
c20,8
c21,8
c22,8
c23,8
c24,8
c25,8
c26,8
c27,8
c28,8
@.
@//E*O*F Release2/Examples/mesh_c.d//
chmod u=r,g=r,o=r Release2/Examples/mesh_c.d
 
echo x - Release2/Examples/mesh_d.d
sed 's/^@//' > "Release2/Examples/mesh_d.d" <<'@//E*O*F Release2/Examples/mesh_d.d//'
#Elt:
 a1, a2, a3, a4, a5, a6, a7, a8, a9,a10,
a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,
a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,
a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,
a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,
a51,a52,a53,a54,a55,
 b1, b2, b3, b4, b5, b6, b7, b8, b9,b10,
b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,
b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,
b41,b42,
 c1, c2, c3, c4, c5, c6, c7, c8, c9,c10,
c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
c21,c22,c23,c24,c25,c26,c27,c28,
 d1, d2, d3, d4, d5, d6, d7, d8, d9,d10,
d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,
d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,
d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,
d51,d52,d53,d54,d55,d56,d57,
 e1, e2, e3, e4, e5, e6, e7, e8, e9,e10,
e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,
e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,
e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,
e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,
e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,
e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,
e81,e82,e83,e84,e85,e86,e87,e88,e89,e90,
e91,e92,e93,e94,e95,e96.
No:
*0,*1,*2,*3,*4,*5,*6,*7,*8,*9,*10,*11,*12.

*long(Elt)
a1
a34
a54
b19
b39
e19
e22
@.
*usual(Elt)
a3
a39
b11
b13
b15
b24
b25
b27
b31
b32
c5
c6
c8
c10
c12
c14
d1
d2
d28
d29
e2
e4
e5
e6
e7
e8
e14
e15
e16
e17
e18
e20
e21
e24
e25
e26
e27
e28
e29
e32
e34
e35
e37
e40
e43
e44
e45
e46
e48
e49
e50
e52
e55
e57
e58
e60
e62
e63
e65
e66
e67
e69
e70
e71
e72
e73
e74
e80
e81
e83
e86
e88
e89
e90
e91
e92
e93
e94
e95
@.
*short(Elt)
a6
a9
a11
a13
a15
a19
a22
a25
a26
a28
a31
a35
a40
a44
b5
b7
b8
b10
b16
b18
b26
b33
b36
b37
c7
c13
d3
d4
d6
d8
d10
d12
d14
d15
d16
d18
d20
d22
d24
d26
d27
e1
e3
e9
e12
e23
e30
e31
e33
e36
e38
e51
e53
e54
e56
e59
e61
e64
e68
e82
e87
@.
*circuit(Elt)
c15
c16
c17
c18
c19
c24
c25
c26
c27
c28
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
@.
*half_circuit(Elt)
a36
a37
a45
a46
a47
a48
a49
a50
a51
a52
a53
b3
b6
b9
b12
b17
b20
b41
b42
@.
*quarter_circuit(Elt)
e75
e76
e77
e78
e84
e85
@.
*short_for_hole(Elt)
a16
a17
a18
a23
a33
b28
b30
b34
b35
c3
c4
e13
e96
@.
*long_for_hole(Elt)
c2
e41
e79
@.
*circuit_hole(Elt)
c20
c21
c22
c23
@.
*half_circuit_hole(Elt)
a38
a42
a43
a55
b1
b14
b22
b29
b38
b40
e10
e11
e39
e47
@.
*not_important(Elt)
a2
a4
a5
a7
a8
a10
a12
a14
a20
a21
a24
a27
a29
a30
a32
a41
b2
b4
b21
b23
c1
c9
c11
d5
d7
d9
d11
d13
d17
d19
d21
d23
d25
e42
@.
*free(Elt)
a39
a40
c6
c7
c11
c12
c13
c15
c16
c17
c18
c19
c20
c27
c28
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d17
d18
d19
d20
d21
d22
d23
d24
d25
d26
d27
d29
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
e6
e16
e19
e20
e22
e23
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e36
e45
e48
e49
e51
e52
e53
e54
e66
e71
e73
e83
e86
e88
e90
e92
e93
e94
@.
*one_side_fixed(Elt)
a34
a35
a41
c2
c3
c5
c8
c10
c14
e5
e7
e15
e17
e18
e21
e24
e37
e44
e46
e50
e65
e67
e69
e74
e76
e77
e78
e82
e84
e85
e89
e91
e95
@.
*two_side_fixed(Elt)
a36
a37
a38
a42
a43
a45
a46
a47
a48
a49
a50
a51
a52
a53
a55
b6
b9
b12
b14
b17
b20
b22
b29
b38
b40
b41
b42
e10
e11
e39
e47
@.
*fixed(Elt)
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24
a25
a26
a27
a28
a29
a30
a31
a32
a33
a44
a54
b1
b2
b3
b4
b5
b7
b8
b10
b11
b13
b15
b16
b18
b19
b21
b23
b24
b25
b26
b27
b28
b30
b31
b32
b33
b34
b35
b36
b37
b39
c1
c4
c9
c21
c22
c23
c24
c25
c26
d1
d2
d28
d30
e1
e2
e3
e4
e8
e9
e12
e13
e14
e25
e38
e40
e41
e42
e43
e55
e56
e57
e58
e59
e60
e61
e62
e63
e64
e68
e70
e72
e75
e79
e80
e81
e87
e96
@.
*not_loaded(Elt)
a1
a2
a3
a4
a5
a6
a7
a23
a24
a25
a26
a27
a28
a29
a33
a36
a37
a42
a44
a45
a46
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12
b13
b14
b15
b16
b17
b18
b19
b20
b21
b23
b24
c1
c2
c3
c4
c5
c6
c7
c8
c9
c15
c17
c18
c20
c21
c22
c23
c26
d1
d2
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d20
d21
d22
d23
d24
d25
d26
d27
d28
d29
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d49
d50
d51
d52
d53
d54
d55
d56
d57
e1
e2
e3
e4
e5
e6
e7
e8
e9
e10
e11
e12
e13
e14
e15
e16
e17
e18
e19
e20
e21
e22
e23
e24
e25
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e39
e40
e41
e42
e43
e44
e45
e46
e47
e48
e49
e50
e51
e52
e53
e54
e55
e56
e57
e58
e62
e63
e64
e65
e66
e67
e68
e69
e70
e71
e72
e73
e74
e77
e78
e79
e80
e81
e82
e83
e84
e85
e86
e87
e88
e89
e90
e91
e92
e93
e94
e95
e96
@.
*one_side_loaded(Elt)
a34
a35
a40
a41
a54
d45
d46
d47
d48
e36
e38
e59
e61
@.
*two_side_loaded(Elt)
e37
@.
*cont_loaded(Elt)
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a30
a31
a32
a38
a39
a43
a47
a48
a49
a50
a51
a52
a53
b22
b25
b26
b27
b28
b29
b30
b31
b32
b33
b34
b35
b36
b37
b38
b39
b40
b41
b42
c10
c11
c12
c13
c14
c16
c19
c24
c25
c27
c28
d17
d18
d19
e60
e75
e76
@.
*neighbour(Elt,Elt)
a1,a2
a1,a44
a10,a11
a10,a49
a11,a12
a11,a49
a12,a13
a12,a51
a13,a14
a14,a15
a14,a53
a15,a16
a15,a38
a16,a17
a16,a55
a17,a18
a18,a19
a18,a38
a19,a20
a2,a3
a2,a54
a20,a21
a20,a52
a21,a22
a22,a23
a22,a50
a23,a24
a24,a1
a24,a42
a25,a26
a25,a35
a26,a27
a26,a36
a27,a28
a27,a37
a28,a29
a28,a37
a29,a30
a29,a46
a3,a4
a30,a31
a30,a46
a31,a32
a31,a48
a32,a33
a32,a48
a33,a25
a34,a35
a34,a40
a35,a26
a35,a40
a36,a27
a36,a4
a37,a5
a37,a6
a38,a16
a38,a19
a39,a41
a4,a34
a4,a5
a40,a39
a41,a44
a42,a25
a44,a24
a44,a54
a45,a28
a45,a29
a46,a7
a46,a8
a47,a30
a47,a31
a48,a10
a48,a9
a49,a32
a49,a43
a5,a36
a5,a6
a50,a11
a50,a12
a51,a22
a52,a14
a53,a20
a54,a1
a54,a44
a55,a17
a55,a18
a6,a45
a6,a7
a7,a45
a7,a8
a8,a47
a8,a9
a9,a10
a9,a47
b1,b2
b1,b34
b10,b11
b10,b9
b11,b12
b11,b15
b12,b15
b12,b16
b13,b12
b13,b16
b14,b13
b15,b14
b15,b30
b16,b17
b16,b24
b17,b18
b18,b8
b18,b9
b19,b18
b19,b20
b2,b3
b2,b5
b20,b21
b20,b24
b21,b22
b21,b25
b22,b23
b22,b25
b23,b19
b23,b20
b24,b17
b24,b21
b25,b26
b25,b38
b26,b27
b26,b38
b27,b28
b27,b42
b28,b13
b29,b27
b3,b4
b3,b5
b30,b31
b31,b29
b31,b32
b32,b33
b32,b42
b33,b34
b33,b41
b34,b2
b34,b40
b35,b1
b35,b36
b36,b37
b36,b40
b37,b39
b37,b41
b38,b37
b39,b22
b39,b23
b4,b35
b40,b33
b40,b35
b41,b32
b41,b36
b42,b26
b42,b31
b5,b10
b5,b6
b6,b10
b6,b7
b7,b3
b7,b4
b8,b6
b8,b7
b9,b11
b9,b8
c1,c2
c1,c23
c10,c11
c10,c25
c11,c12
c11,c27
c12,c13
c12,c28
c13,c14
c13,c28
c14,c1
c14,c24
c15,c8
c16,c10
c16,c11
c17,c7
c18,c6
c19,c13
c19,c14
c2,c23
c2,c3
c20,c2
c20,c3
c21,c4
c21,c5
c24,c1
c26,c9
c27,c12
c3,c22
c3,c4
c4,c22
c4,c5
c5,c18
c5,c6
c6,c17
c6,c7
c7,c15
c7,c8
c8,c26
c8,c9
c9,c10
c9,c25
d1,d2
d1,d30
d10,d11
d10,d39
d11,d12
d11,d40
d12,d13
d12,d41
d13,d14
d13,d42
d14,d15
d14,d43
d15,d16
d15,d44
d16,d17
d16,d45
d17,d18
d17,d46
d18,d19
d18,d47
d19,d20
d19,d48
d2,d3
d2,d31
d20,d21
d20,d49
d21,d22
d21,d50
d22,d23
d22,d51
d23,d24
d23,d52
d24,d25
d24,d53
d25,d26
d25,d54
d26,d27
d26,d55
d27,d28
d27,d56
d28,d29
d28,d57
d3,d32
d3,d4
d30,d2
d31,d3
d32,d4
d33,d5
d34,d6
d35,d7
d36,d8
d37,d9
d38,d10
d39,d11
d4,d33
d4,d5
d40,d12
d41,d13
d42,d14
d43,d15
d44,d16
d45,d17
d46,d18
d47,d19
d48,d20
d49,d21
d5,d34
d5,d6
d50,d22
d51,d23
d52,d24
d53,d25
d54,d26
d55,d27
d56,d28
d57,d29
d6,d35
d6,d7
d7,d36
d7,d8
d8,d37
d8,d9
d9,d10
d9,d38
e1,e2
e1,e5
e10,e95
e10,e96
e11,e13
e11,e3
e12,e11
e12,e14
e13,e10
e13,e12
e14,e15
e14,e40
e15,e16
e15,e40
e16,e17
e16,e28
e17,e12
e17,e14
e18,e19
e18,e20
e19,e20
e19,e53
e2,e18
e2,e3
e20,e22
e21,e1
e22,e21
e22,e26
e23,e22
e23,e27
e24,e23
e24,e25
e25,e55
e25,e85
e26,e23
e26,e94
e27,e24
e27,e51
e28,e17
e28,e52
e29,e15
e29,e16
e3,e18
e3,e96
e30,e29
e30,e86
e31,e30
e31,e86
e32,e31
e32,e84
e33,e32
e33,e84
e34,e33
e34,e78
e35,e34
e36,e35
e36,e76
e37,e36
e37,e76
e38,e37
e38,e75
e39,e38
e39,e41
e4,e1
e4,e5
e40,e39
e40,e41
e41,e42
e41,e47
e42,e43
e43,e44
e43,e87
e44,e45
e44,e49
e45,e46
e45,e48
e46,e43
e46,e47
e47,e79
e47,e91
e48,e46
e48,e92
e49,e45
e49,e69
e5,e6
e5,e93
e50,e71
e51,e54
e52,e29
e52,e30
e53,e28
e53,e52
e54,e26
e54,e94
e55,e56
e55,e85
e56,e57
e56,e78
e57,e58
e58,e59
e58,e77
e59,e60
e59,e77
e6,e7
e6,e93
e60,e61
e60,e75
e61,e62
e61,e72
e62,e63
e62,e72
e63,e64
e63,e65
e64,e24
e64,e25
e65,e64
e65,e73
e66,e65
e66,e67
e67,e66
e67,e70
e68,e67
e68,e81
e69,e68
e69,e81
e7,e8
e7,e87
e70,e50
e70,e68
e71,e73
e71,e74
e72,e50
e72,e70
e73,e66
e73,e74
e74,e62
e74,e63
e75,e37
e75,e61
e76,e59
e76,e60
e77,e35
e77,e36
e78,e33
e78,e57
e79,e38
e79,e39
e8,e89
e8,e9
e80,e79
e80,e91
e81,e80
e81,e82
e82,e80
e82,e83
e83,e49
e83,e69
e84,e55
e84,e56
e85,e31
e85,e32
e86,e51
e86,e54
e87,e44
e87,e8
e88,e89
e88,e90
e89,e9
e89,e90
e9,e10
e9,e13
e90,e6
e90,e7
e91,e48
e92,e82
e92,e83
e93,e88
e93,e95
e94,e19
e94,e53
e95,e4
e95,e88
e96,e11
e96,e4
@.
*opposite_r(Elt,Elt)
a11,a3
a9,a3
a31,a25
a13,a1
a15,a1
a17,a1
a19,a1
a22,a1
a23,a1
a32,a22
a33,a23
a34,a54
a37,a45
a39,a42
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b13,b27
b15,b31
b10,b33
b8,b36
b5,b34
b7,b35
b18,b37
b16,b26
b11,b32
b19,b39
b24,b25
c6,c12
c2,c14
c3,c5
c8,c10
c10,c14
c11,c7
c13,c5
c15,c16
c27,c17
c28,c18
c20,c19
c21,c22
c23,c24
c25,c26
d5,d7
d9,d11
d13,d15
d17,d19
d21,d23
d25,d27
e4,e2
e96,e2
e8,e43
e13,e43
e41,e14
e83,e45
e90,e93
e81,e70
e63,e25
e62,e60
e57,e55
e32,e34
@.
*equal_r(Elt,Elt)
a16,a18
a29,a7
a31,a9
a33,a23
a34,a54
a36,a37
a37,a45
a38,a55
a42,a43
a45,a46
a46,a47
a47,a48
a48,a49
a49,a50
a50,a51
a51,a52
a52,a53
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b25,b39
b38,b22
b20,b17
b3,b6
b9,b12
b29,b14
b40,b1
b42,b41
c6,c12
c2,c14
c10,c14
c15,c17
c17,c18
c18,c21
c21,c22
c22,c20
c20,c23
c23,c24
c19,c20
c28,c19
c27,c28
c16,c27
c25,c16
c26,c25
d5,d7
d9,d11
d17,d19
d21,d23
d30,d31
d31,d32
d32,d33
d34,d35
d36,d37
d38,d39
d40,d41
d42,d43
d44,d45
d46,d47
d48,d49
d50,d51
d52,d53
d54,d55
d55,d56
d56,d57
e90,e93
e93,e4
e13,e96
e11,e10
e41,e79
e39,e47
e75,e76
e76,e77
e77,e78
e78,e84
e84,e85
e14,e16
e2,e20
e18,e21
e60,e37
e59,e36
e35,e58
e57,e34
e33,e56
@.
mesh(Elt, No)
a2,1
a3,8
a4,1
a5,1
a6,2
a7,1
a8,1
a9,3
a10,1
a11,3
a12,1
a13,1
a14,1
a15,4
a16,1
a17,2
a18,1
a19,4
a20,1
a21,1
a22,2
a23,2
a24,1
a25,2
a26,1
a27,1
a28,2
a29,1
a30,1
a31,3
a32,2
a33,2
a34,11
a35,1
a36,12
a37,12
a38,12
a39,5
a40,2
a41,1
a42,5
a43,5
a44,1
a45,12
a46,12
a47,12
a48,12
a49,12
a50,12
a51,12
a52,12
a53,12
a54,11
a55,12
b1,6
b2,1
b3,6
b4,1
b5,1
b6,6
b7,1
b8,2
b9,6
b10,2
b11,6
b12,6
b13,3
b14,6
b15,3
b16,3
b17,8
b18,3
b19,7
b20,8
b21,1
b22,8
b23,1
b24,7
b25,7
b26,2
b27,2
b28,1
b29,6
b30,1
b31,2
b32,4
b33,2
b34,2
b35,2
b36,2
b37,1
b38,8
b39,7
b40,6
b41,6
b42,6
c1,1
c2,2
c3,1
c4,1
c5,3
c6,2
c7,2
c8,3
c9,1
c10,2
c11,1
c12,2
c13,1
c14,2
c15,8
c16,8
c17,8
c18,8
c19,8
c20,8
c21,8
c22,8
c23,8
c24,8
c25,8
c26,8
c27,8
c28,8
e1,1
e2,3
e3,1
e4,2
e5,2
e6,3
e7,2
e8,2
e9,1
e10,4
e11,4
e12,1
e13,1
e14,5
e15,2
e16,5
e17,2
e18,3
e19,10
e20,3
e21,3
e22,12
e23,2
e24,4
e25,2
e26,3
e27,3
e28,5
e29,2
e30,1
e31,1
e32,2
e33,1
e34,2
e35,1
e36,1
e37,2
e38,1
e39,6
e40,2
e41,6
e42,1
e43,5
e44,2
e45,5
e46,2
e47,6
e48,2
e49,5
e50,2
e51,1
e52,3
e53,1
e54,1
e55,2
e56,1
e57,2
e58,1
e59,1
e60,2
e61,1
e62,2
e63,3
e64,1
e65,4
e66,2
e67,2
e68,1
e69,2
e70,2
e71,2
e72,3
e73,2
e74,4
e75,9
e76,9
e77,9
e78,9
e79,6
e80,5
e81,2
e82,2
e83,2
e84,9
e85,9
e86,5
e87,1
e88,3
e89,2
e90,2
e91,2
e92,5
e93,2
e94,2
e95,2
e96,1
@.

mesh
d1,2
d2,4
d3,1
d4,1
d5,1
d6,2
d7,1
d8,2
d9,1
d10,2
d11,1
d12,2
d13,1
d14,2
d15,2
d16,2
d17,1
d18,2
d19,1
d20,2
d21,1
d22,2
d23,1
d24,1
d25,1
d26,1
d27,2
d28,4
d29,2
d30,12
d31,12
d32,12
d33,12
d34,12
d35,12
d36,12
d37,12
d38,12
d39,12
d40,12
d41,12
d42,12
d43,12
d44,12
d45,12
d46,12
d47,12
d48,12
d49,12
d50,12
d51,12
d52,12
d53,12
d54,12
d55,12
d56,12
d57,12
@.
@//E*O*F Release2/Examples/mesh_d.d//
chmod u=r,g=r,o=r Release2/Examples/mesh_d.d
 
echo x - Release2/Examples/mesh_e.d
sed 's/^@//' > "Release2/Examples/mesh_e.d" <<'@//E*O*F Release2/Examples/mesh_e.d//'
#Elt:
 a1, a2, a3, a4, a5, a6, a7, a8, a9,a10,
a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,
a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,
a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,
a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,
a51,a52,a53,a54,a55,
 b1, b2, b3, b4, b5, b6, b7, b8, b9,b10,
b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,
b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,
b41,b42,
 c1, c2, c3, c4, c5, c6, c7, c8, c9,c10,
c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
c21,c22,c23,c24,c25,c26,c27,c28,
 d1, d2, d3, d4, d5, d6, d7, d8, d9,d10,
d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,
d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,
d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,
d51,d52,d53,d54,d55,d56,d57,
 e1, e2, e3, e4, e5, e6, e7, e8, e9,e10,
e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,
e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,
e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,
e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,
e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,
e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,
e81,e82,e83,e84,e85,e86,e87,e88,e89,e90,
e91,e92,e93,e94,e95,e96.
No:
*0,*1,*2,*3,*4,*5,*6,*7,*8,*9,*10,*11,*12.

*long(Elt)
a1
a34
a54
b19
b39
e19
e22
@.
*usual(Elt)
a3
a39
b11
b13
b15
b24
b25
b27
b31
b32
c5
c6
c8
c10
c12
c14
d1
d2
d28
d29
e2
e4
e5
e6
e7
e8
e14
e15
e16
e17
e18
e20
e21
e24
e25
e26
e27
e28
e29
e32
e34
e35
e37
e40
e43
e44
e45
e46
e48
e49
e50
e52
e55
e57
e58
e60
e62
e63
e65
e66
e67
e69
e70
e71
e72
e73
e74
e80
e81
e83
e86
e88
e89
e90
e91
e92
e93
e94
e95
@.
*short(Elt)
a6
a9
a11
a13
a15
a19
a22
a25
a26
a28
a31
a35
a40
a44
b5
b7
b8
b10
b16
b18
b26
b33
b36
b37
c7
c13
d3
d4
d6
d8
d10
d12
d14
d15
d16
d18
d20
d22
d24
d26
d27
e1
e3
e9
e12
e23
e30
e31
e33
e36
e38
e51
e53
e54
e56
e59
e61
e64
e68
e82
e87
@.
*circuit(Elt)
c15
c16
c17
c18
c19
c24
c25
c26
c27
c28
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
@.
*half_circuit(Elt)
a36
a37
a45
a46
a47
a48
a49
a50
a51
a52
a53
b3
b6
b9
b12
b17
b20
b41
b42
@.
*quarter_circuit(Elt)
e75
e76
e77
e78
e84
e85
@.
*short_for_hole(Elt)
a16
a17
a18
a23
a33
b28
b30
b34
b35
c3
c4
e13
e96
@.
*long_for_hole(Elt)
c2
e41
e79
@.
*circuit_hole(Elt)
c20
c21
c22
c23
@.
*half_circuit_hole(Elt)
a38
a42
a43
a55
b1
b14
b22
b29
b38
b40
e10
e11
e39
e47
@.
*not_important(Elt)
a2
a4
a5
a7
a8
a10
a12
a14
a20
a21
a24
a27
a29
a30
a32
a41
b2
b4
b21
b23
c1
c9
c11
d5
d7
d9
d11
d13
d17
d19
d21
d23
d25
e42
@.
*free(Elt)
a39
a40
c6
c7
c11
c12
c13
c15
c16
c17
c18
c19
c20
c27
c28
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d17
d18
d19
d20
d21
d22
d23
d24
d25
d26
d27
d29
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d45
d46
d47
d48
d49
d50
d51
d52
d53
d54
d55
d56
d57
e6
e16
e19
e20
e22
e23
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e36
e45
e48
e49
e51
e52
e53
e54
e66
e71
e73
e83
e86
e88
e90
e92
e93
e94
@.
*one_side_fixed(Elt)
a34
a35
a41
c2
c3
c5
c8
c10
c14
e5
e7
e15
e17
e18
e21
e24
e37
e44
e46
e50
e65
e67
e69
e74
e76
e77
e78
e82
e84
e85
e89
e91
e95
@.
*two_side_fixed(Elt)
a36
a37
a38
a42
a43
a45
a46
a47
a48
a49
a50
a51
a52
a53
a55
b6
b9
b12
b14
b17
b20
b22
b29
b38
b40
b41
b42
e10
e11
e39
e47
@.
*fixed(Elt)
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24
a25
a26
a27
a28
a29
a30
a31
a32
a33
a44
a54
b1
b2
b3
b4
b5
b7
b8
b10
b11
b13
b15
b16
b18
b19
b21
b23
b24
b25
b26
b27
b28
b30
b31
b32
b33
b34
b35
b36
b37
b39
c1
c4
c9
c21
c22
c23
c24
c25
c26
d1
d2
d28
d30
e1
e2
e3
e4
e8
e9
e12
e13
e14
e25
e38
e40
e41
e42
e43
e55
e56
e57
e58
e59
e60
e61
e62
e63
e64
e68
e70
e72
e75
e79
e80
e81
e87
e96
@.
*not_loaded(Elt)
a1
a2
a3
a4
a5
a6
a7
a23
a24
a25
a26
a27
a28
a29
a33
a36
a37
a42
a44
a45
a46
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12
b13
b14
b15
b16
b17
b18
b19
b20
b21
b23
b24
c1
c2
c3
c4
c5
c6
c7
c8
c9
c15
c17
c18
c20
c21
c22
c23
c26
d1
d2
d3
d4
d5
d6
d7
d8
d9
d10
d11
d12
d13
d14
d15
d16
d20
d21
d22
d23
d24
d25
d26
d27
d28
d29
d30
d31
d32
d33
d34
d35
d36
d37
d38
d39
d40
d41
d42
d43
d44
d49
d50
d51
d52
d53
d54
d55
d56
d57
e1
e2
e3
e4
e5
e6
e7
e8
e9
e10
e11
e12
e13
e14
e15
e16
e17
e18
e19
e20
e21
e22
e23
e24
e25
e26
e27
e28
e29
e30
e31
e32
e33
e34
e35
e39
e40
e41
e42
e43
e44
e45
e46
e47
e48
e49
e50
e51
e52
e53
e54
e55
e56
e57
e58
e62
e63
e64
e65
e66
e67
e68
e69
e70
e71
e72
e73
e74
e77
e78
e79
e80
e81
e82
e83
e84
e85
e86
e87
e88
e89
e90
e91
e92
e93
e94
e95
e96
@.
*one_side_loaded(Elt)
a34
a35
a40
a41
a54
d45
d46
d47
d48
e36
e38
e59
e61
@.
*two_side_loaded(Elt)
e37
@.
*cont_loaded(Elt)
a8
a9
a10
a11
a12
a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a30
a31
a32
a38
a39
a43
a47
a48
a49
a50
a51
a52
a53
b22
b25
b26
b27
b28
b29
b30
b31
b32
b33
b34
b35
b36
b37
b38
b39
b40
b41
b42
c10
c11
c12
c13
c14
c16
c19
c24
c25
c27
c28
d17
d18
d19
e60
e75
e76
@.
*neighbour(Elt,Elt)
a1,a2
a1,a44
a10,a11
a10,a49
a11,a12
a11,a49
a12,a13
a12,a51
a13,a14
a14,a15
a14,a53
a15,a16
a15,a38
a16,a17
a16,a55
a17,a18
a18,a19
a18,a38
a19,a20
a2,a3
a2,a54
a20,a21
a20,a52
a21,a22
a22,a23
a22,a50
a23,a24
a24,a1
a24,a42
a25,a26
a25,a35
a26,a27
a26,a36
a27,a28
a27,a37
a28,a29
a28,a37
a29,a30
a29,a46
a3,a4
a30,a31
a30,a46
a31,a32
a31,a48
a32,a33
a32,a48
a33,a25
a34,a35
a34,a40
a35,a26
a35,a40
a36,a27
a36,a4
a37,a5
a37,a6
a38,a16
a38,a19
a39,a41
a4,a34
a4,a5
a40,a39
a41,a44
a42,a25
a44,a24
a44,a54
a45,a28
a45,a29
a46,a7
a46,a8
a47,a30
a47,a31
a48,a10
a48,a9
a49,a32
a49,a43
a5,a36
a5,a6
a50,a11
a50,a12
a51,a22
a52,a14
a53,a20
a54,a1
a54,a44
a55,a17
a55,a18
a6,a45
a6,a7
a7,a45
a7,a8
a8,a47
a8,a9
a9,a10
a9,a47
b1,b2
b1,b34
b10,b11
b10,b9
b11,b12
b11,b15
b12,b15
b12,b16
b13,b12
b13,b16
b14,b13
b15,b14
b15,b30
b16,b17
b16,b24
b17,b18
b18,b8
b18,b9
b19,b18
b19,b20
b2,b3
b2,b5
b20,b21
b20,b24
b21,b22
b21,b25
b22,b23
b22,b25
b23,b19
b23,b20
b24,b17
b24,b21
b25,b26
b25,b38
b26,b27
b26,b38
b27,b28
b27,b42
b28,b13
b29,b27
b3,b4
b3,b5
b30,b31
b31,b29
b31,b32
b32,b33
b32,b42
b33,b34
b33,b41
b34,b2
b34,b40
b35,b1
b35,b36
b36,b37
b36,b40
b37,b39
b37,b41
b38,b37
b39,b22
b39,b23
b4,b35
b40,b33
b40,b35
b41,b32
b41,b36
b42,b26
b42,b31
b5,b10
b5,b6
b6,b10
b6,b7
b7,b3
b7,b4
b8,b6
b8,b7
b9,b11
b9,b8
c1,c2
c1,c23
c10,c11
c10,c25
c11,c12
c11,c27
c12,c13
c12,c28
c13,c14
c13,c28
c14,c1
c14,c24
c15,c8
c16,c10
c16,c11
c17,c7
c18,c6
c19,c13
c19,c14
c2,c23
c2,c3
c20,c2
c20,c3
c21,c4
c21,c5
c24,c1
c26,c9
c27,c12
c3,c22
c3,c4
c4,c22
c4,c5
c5,c18
c5,c6
c6,c17
c6,c7
c7,c15
c7,c8
c8,c26
c8,c9
c9,c10
c9,c25
d1,d2
d1,d30
d10,d11
d10,d39
d11,d12
d11,d40
d12,d13
d12,d41
d13,d14
d13,d42
d14,d15
d14,d43
d15,d16
d15,d44
d16,d17
d16,d45
d17,d18
d17,d46
d18,d19
d18,d47
d19,d20
d19,d48
d2,d3
d2,d31
d20,d21
d20,d49
d21,d22
d21,d50
d22,d23
d22,d51
d23,d24
d23,d52
d24,d25
d24,d53
d25,d26
d25,d54
d26,d27
d26,d55
d27,d28
d27,d56
d28,d29
d28,d57
d3,d32
d3,d4
d30,d2
d31,d3
d32,d4
d33,d5
d34,d6
d35,d7
d36,d8
d37,d9
d38,d10
d39,d11
d4,d33
d4,d5
d40,d12
d41,d13
d42,d14
d43,d15
d44,d16
d45,d17
d46,d18
d47,d19
d48,d20
d49,d21
d5,d34
d5,d6
d50,d22
d51,d23
d52,d24
d53,d25
d54,d26
d55,d27
d56,d28
d57,d29
d6,d35
d6,d7
d7,d36
d7,d8
d8,d37
d8,d9
d9,d10
d9,d38
e1,e2
e1,e5
e10,e95
e10,e96
e11,e13
e11,e3
e12,e11
e12,e14
e13,e10
e13,e12
e14,e15
e14,e40
e15,e16
e15,e40
e16,e17
e16,e28
e17,e12
e17,e14
e18,e19
e18,e20
e19,e20
e19,e53
e2,e18
e2,e3
e20,e22
e21,e1
e22,e21
e22,e26
e23,e22
e23,e27
e24,e23
e24,e25
e25,e55
e25,e85
e26,e23
e26,e94
e27,e24
e27,e51
e28,e17
e28,e52
e29,e15
e29,e16
e3,e18
e3,e96
e30,e29
e30,e86
e31,e30
e31,e86
e32,e31
e32,e84
e33,e32
e33,e84
e34,e33
e34,e78
e35,e34
e36,e35
e36,e76
e37,e36
e37,e76
e38,e37
e38,e75
e39,e38
e39,e41
e4,e1
e4,e5
e40,e39
e40,e41
e41,e42
e41,e47
e42,e43
e43,e44
e43,e87
e44,e45
e44,e49
e45,e46
e45,e48
e46,e43
e46,e47
e47,e79
e47,e91
e48,e46
e48,e92
e49,e45
e49,e69
e5,e6
e5,e93
e50,e71
e51,e54
e52,e29
e52,e30
e53,e28
e53,e52
e54,e26
e54,e94
e55,e56
e55,e85
e56,e57
e56,e78
e57,e58
e58,e59
e58,e77
e59,e60
e59,e77
e6,e7
e6,e93
e60,e61
e60,e75
e61,e62
e61,e72
e62,e63
e62,e72
e63,e64
e63,e65
e64,e24
e64,e25
e65,e64
e65,e73
e66,e65
e66,e67
e67,e66
e67,e70
e68,e67
e68,e81
e69,e68
e69,e81
e7,e8
e7,e87
e70,e50
e70,e68
e71,e73
e71,e74
e72,e50
e72,e70
e73,e66
e73,e74
e74,e62
e74,e63
e75,e37
e75,e61
e76,e59
e76,e60
e77,e35
e77,e36
e78,e33
e78,e57
e79,e38
e79,e39
e8,e89
e8,e9
e80,e79
e80,e91
e81,e80
e81,e82
e82,e80
e82,e83
e83,e49
e83,e69
e84,e55
e84,e56
e85,e31
e85,e32
e86,e51
e86,e54
e87,e44
e87,e8
e88,e89
e88,e90
e89,e9
e89,e90
e9,e10
e9,e13
e90,e6
e90,e7
e91,e48
e92,e82
e92,e83
e93,e88
e93,e95
e94,e19
e94,e53
e95,e4
e95,e88
e96,e11
e96,e4
@.
*opposite_r(Elt,Elt)
a11,a3
a9,a3
a31,a25
a13,a1
a15,a1
a17,a1
a19,a1
a22,a1
a23,a1
a32,a22
a33,a23
a34,a54
a37,a45
a39,a42
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b13,b27
b15,b31
b10,b33
b8,b36
b5,b34
b7,b35
b18,b37
b16,b26
b11,b32
b19,b39
b24,b25
c6,c12
c2,c14
c3,c5
c8,c10
c10,c14
c11,c7
c13,c5
c15,c16
c27,c17
c28,c18
c20,c19
c21,c22
c23,c24
c25,c26
d5,d7
d9,d11
d13,d15
d17,d19
d21,d23
d25,d27
e4,e2
e96,e2
e8,e43
e13,e43
e41,e14
e83,e45
e90,e93
e81,e70
e63,e25
e62,e60
e57,e55
e32,e34
@.
*equal_r(Elt,Elt)
a16,a18
a29,a7
a31,a9
a33,a23
a34,a54
a36,a37
a37,a45
a38,a55
a42,a43
a45,a46
a46,a47
a47,a48
a48,a49
a49,a50
a50,a51
a51,a52
a52,a53
b1,b3
b6,b40
b41,b9
b12,b42
b17,b38
b22,b20
b29,b42
b14,b12
b25,b39
b38,b22
b20,b17
b3,b6
b9,b12
b29,b14
b40,b1
b42,b41
c6,c12
c2,c14
c10,c14
c15,c17
c17,c18
c18,c21
c21,c22
c22,c20
c20,c23
c23,c24
c19,c20
c28,c19
c27,c28
c16,c27
c25,c16
c26,c25
d5,d7
d9,d11
d17,d19
d21,d23
d30,d31
d31,d32
d32,d33
d34,d35
d36,d37
d38,d39
d40,d41
d42,d43
d44,d45
d46,d47
d48,d49
d50,d51
d52,d53
d54,d55
d55,d56
d56,d57
e90,e93
e93,e4
e13,e96
e11,e10
e41,e79
e39,e47
e75,e76
e76,e77
e77,e78
e78,e84
e84,e85
e14,e16
e2,e20
e18,e21
e60,e37
e59,e36
e35,e58
e57,e34
e33,e56
@.
mesh(Elt, No)
a2,1
a3,8
a4,1
a5,1
a6,2
a7,1
a8,1
a9,3
a10,1
a11,3
a12,1
a13,1
a14,1
a15,4
a16,1
a17,2
a18,1
a19,4
a20,1
a21,1
a22,2
a23,2
a24,1
a25,2
a26,1
a27,1
a28,2
a29,1
a30,1
a31,3
a32,2
a33,2
a34,11
a35,1
a36,12
a37,12
a38,12
a39,5
a40,2
a41,1
a42,5
a43,5
a44,1
a45,12
a46,12
a47,12
a48,12
a49,12
a50,12
a51,12
a52,12
a53,12
a54,11
a55,12
b1,6
b2,1
b3,6
b4,1
b5,1
b6,6
b7,1
b8,2
b9,6
b10,2
b11,6
b12,6
b13,3
b14,6
b15,3
b16,3
b17,8
b18,3
b19,7
b20,8
b21,1
b22,8
b23,1
b24,7
b25,7
b26,2
b27,2
b28,1
b29,6
b30,1
b31,2
b32,4
b33,2
b34,2
b35,2
b36,2
b37,1
b38,8
b39,7
b40,6
b41,6
b42,6
c1,1
c2,2
c3,1
c4,1
c5,3
c6,2
c7,2
c8,3
c9,1
c10,2
c11,1
c12,2
c13,1
c14,2
c15,8
c16,8
c17,8
c18,8
c19,8
c20,8
c21,8
c22,8
c23,8
c24,8
c25,8
c26,8
c27,8
c28,8
d1,2
d2,4
d3,1
d4,1
d5,1
d6,2
d7,1
d8,2
d9,1
d10,2
d11,1
d12,2
d13,1
d14,2
d15,2
d16,2
d17,1
d18,2
d19,1
d20,2
d21,1
d22,2
d23,1
d24,1
d25,1
d26,1
d27,2
d28,4
d29,2
d30,12
d31,12
d32,12
d33,12
d34,12
d35,12
d36,12
d37,12
d38,12
d39,12
d40,12
d41,12
d42,12
d43,12
d44,12
d45,12
d46,12
d47,12
d48,12
d49,12
d50,12
d51,12
d52,12
d53,12
d54,12
d55,12
d56,12
d57,12
@.

mesh
e1,1
e2,3
e3,1
e4,2
e5,2
e6,3
e7,2
e8,2
e9,1
e10,4
e11,4
e12,1
e13,1
e14,5
e15,2
e16,5
e17,2
e18,3
e19,10
e20,3
e21,3
e22,12
e23,2
e24,4
e25,2
e26,3
e27,3
e28,5
e29,2
e30,1
e31,1
e32,2
e33,1
e34,2
e35,1
e36,1
e37,2
e38,1
e39,6
e40,2
e41,6
e42,1
e43,5
e44,2
e45,5
e46,2
e47,6
e48,2
e49,5
e50,2
e51,1
e52,3
e53,1
e54,1
e55,2
e56,1
e57,2
e58,1
e59,1
e60,2
e61,1
e62,2
e63,3
e64,1
e65,4
e66,2
e67,2
e68,1
e69,2
e70,2
e71,2
e72,3
e73,2
e74,4
e75,9
e76,9
e77,9
e78,9
e79,6
e80,5
e81,2
e82,2
e83,2
e84,9
e85,9
e86,5
e87,1
e88,3
e89,2
e90,2
e91,2
e92,5
e93,2
e94,2
e95,2
e96,1
@.
@//E*O*F Release2/Examples/mesh_e.d//
chmod u=r,g=r,o=r Release2/Examples/mesh_e.d
 
echo x - Release2/Examples/qs44.d
sed 's/^@//' > "Release2/Examples/qs44.d" <<'@//E*O*F Release2/Examples/qs44.d//'
*E: 0,1,2,3.
*L: *[],[0],[1],[2],[3],[01],[02],[03],[10],[12],[13],[20],[21],[23],[30],[31],[32],[012],[013],[021],[023],[031],[032],[102],[103],[120],[123],[130],[132],[201],[203],[210],[213],[230],[231],[301],[302],[310],[312],[320],[321],[0123],[0132],[0213],[0231],[0312],[0321],[1023],[1032],[1203],[1230],[1302],[1320],[2013],[2031],[2103],[2130],[2301],[2310],[3012],[3021],[3102],[3120],[3201],[3210].

*elt(E)
0
1
2
3
@.
*append(L,L,L) ##-
[], [], []
[], [0], [0]
[0], [], [0]
[], [1], [1]
[1], [], [1]
[], [2], [2]
[2], [], [2]
[], [3], [3]
[3], [], [3]
[], [01], [01]
[0], [1], [01]
[01], [], [01]
[], [02], [02]
[0], [2], [02]
[02], [], [02]
[], [03], [03]
[0], [3], [03]
[03], [], [03]
[], [10], [10]
[1], [0], [10]
[10], [], [10]
[], [12], [12]
[1], [2], [12]
[12], [], [12]
[], [13], [13]
[1], [3], [13]
[13], [], [13]
[], [20], [20]
[2], [0], [20]
[20], [], [20]
[], [21], [21]
[2], [1], [21]
[21], [], [21]
[], [23], [23]
[2], [3], [23]
[23], [], [23]
[], [30], [30]
[3], [0], [30]
[30], [], [30]
[], [31], [31]
[3], [1], [31]
[31], [], [31]
[], [32], [32]
[3], [2], [32]
[32], [], [32]
[], [012], [012]
[0], [12], [012]
[01], [2], [012]
[012], [], [012]
[], [013], [013]
[0], [13], [013]
[01], [3], [013]
[013], [], [013]
[], [021], [021]
[0], [21], [021]
[02], [1], [021]
[021], [], [021]
[], [023], [023]
[0], [23], [023]
[02], [3], [023]
[023], [], [023]
[], [031], [031]
[0], [31], [031]
[03], [1], [031]
[031], [], [031]
[], [032], [032]
[0], [32], [032]
[03], [2], [032]
[032], [], [032]
[], [102], [102]
[1], [02], [102]
[10], [2], [102]
[102], [], [102]
[], [103], [103]
[1], [03], [103]
[10], [3], [103]
[103], [], [103]
[], [120], [120]
[1], [20], [120]
[12], [0], [120]
[120], [], [120]
[], [123], [123]
[1], [23], [123]
[12], [3], [123]
[123], [], [123]
[], [130], [130]
[1], [30], [130]
[13], [0], [130]
[130], [], [130]
[], [132], [132]
[1], [32], [132]
[13], [2], [132]
[132], [], [132]
[], [201], [201]
[2], [01], [201]
[20], [1], [201]
[201], [], [201]
[], [203], [203]
[2], [03], [203]
[20], [3], [203]
[203], [], [203]
[], [210], [210]
[2], [10], [210]
[21], [0], [210]
[210], [], [210]
[], [213], [213]
[2], [13], [213]
[21], [3], [213]
[213], [], [213]
[], [230], [230]
[2], [30], [230]
[23], [0], [230]
[230], [], [230]
[], [231], [231]
[2], [31], [231]
[23], [1], [231]
[231], [], [231]
[], [301], [301]
[3], [01], [301]
[30], [1], [301]
[301], [], [301]
[], [302], [302]
[3], [02], [302]
[30], [2], [302]
[302], [], [302]
[], [310], [310]
[3], [10], [310]
[31], [0], [310]
[310], [], [310]
[], [312], [312]
[3], [12], [312]
[31], [2], [312]
[312], [], [312]
[], [320], [320]
[3], [20], [320]
[32], [0], [320]
[320], [], [320]
[], [321], [321]
[3], [21], [321]
[32], [1], [321]
[321], [], [321]
[], [0123], [0123]
[0], [123], [0123]
[01], [23], [0123]
[012], [3], [0123]
[0123], [], [0123]
[], [0132], [0132]
[0], [132], [0132]
[01], [32], [0132]
[013], [2], [0132]
[0132], [], [0132]
[], [0213], [0213]
[0], [213], [0213]
[02], [13], [0213]
[021], [3], [0213]
[0213], [], [0213]
[], [0231], [0231]
[0], [231], [0231]
[02], [31], [0231]
[023], [1], [0231]
[0231], [], [0231]
[], [0312], [0312]
[0], [312], [0312]
[03], [12], [0312]
[031], [2], [0312]
[0312], [], [0312]
[], [0321], [0321]
[0], [321], [0321]
[03], [21], [0321]
[032], [1], [0321]
[0321], [], [0321]
[], [1023], [1023]
[1], [023], [1023]
[10], [23], [1023]
[102], [3], [1023]
[1023], [], [1023]
[], [1032], [1032]
[1], [032], [1032]
[10], [32], [1032]
[103], [2], [1032]
[1032], [], [1032]
[], [1203], [1203]
[1], [203], [1203]
[12], [03], [1203]
[120], [3], [1203]
[1203], [], [1203]
[], [1230], [1230]
[1], [230], [1230]
[12], [30], [1230]
[123], [0], [1230]
[1230], [], [1230]
[], [1302], [1302]
[1], [302], [1302]
[13], [02], [1302]
[130], [2], [1302]
[1302], [], [1302]
[], [1320], [1320]
[1], [320], [1320]
[13], [20], [1320]
[132], [0], [1320]
[1320], [], [1320]
[], [2013], [2013]
[2], [013], [2013]
[20], [13], [2013]
[201], [3], [2013]
[2013], [], [2013]
[], [2031], [2031]
[2], [031], [2031]
[20], [31], [2031]
[203], [1], [2031]
[2031], [], [2031]
[], [2103], [2103]
[2], [103], [2103]
[21], [03], [2103]
[210], [3], [2103]
[2103], [], [2103]
[], [2130], [2130]
[2], [130], [2130]
[21], [30], [2130]
[213], [0], [2130]
[2130], [], [2130]
[], [2301], [2301]
[2], [301], [2301]
[23], [01], [2301]
[230], [1], [2301]
[2301], [], [2301]
[], [2310], [2310]
[2], [310], [2310]
[23], [10], [2310]
[231], [0], [2310]
[2310], [], [2310]
[], [3012], [3012]
[3], [012], [3012]
[30], [12], [3012]
[301], [2], [3012]
[3012], [], [3012]
[], [3021], [3021]
[3], [021], [3021]
[30], [21], [3021]
[302], [1], [3021]
[3021], [], [3021]
[], [3102], [3102]
[3], [102], [3102]
[31], [02], [3102]
[310], [2], [3102]
[3102], [], [3102]
[], [3120], [3120]
[3], [120], [3120]
[31], [20], [3120]
[312], [0], [3120]
[3120], [], [3120]
[], [3201], [3201]
[3], [201], [3201]
[32], [01], [3201]
[320], [1], [3201]
[3201], [], [3201]
[], [3210], [3210]
[3], [210], [3210]
[32], [10], [3210]
[321], [0], [3210]
[3210], [], [3210]
@.
*components(L,E,L) #--/-##
[0], 0, []
[1], 1, []
[2], 2, []
[3], 3, []
[01], 0, [1]
[02], 0, [2]
[03], 0, [3]
[10], 1, [0]
[12], 1, [2]
[13], 1, [3]
[20], 2, [0]
[21], 2, [1]
[23], 2, [3]
[30], 3, [0]
[31], 3, [1]
[32], 3, [2]
[012], 0, [12]
[013], 0, [13]
[021], 0, [21]
[023], 0, [23]
[031], 0, [31]
[032], 0, [32]
[102], 1, [02]
[103], 1, [03]
[120], 1, [20]
[123], 1, [23]
[130], 1, [30]
[132], 1, [32]
[201], 2, [01]
[203], 2, [03]
[210], 2, [10]
[213], 2, [13]
[230], 2, [30]
[231], 2, [31]
[301], 3, [01]
[302], 3, [02]
[310], 3, [10]
[312], 3, [12]
[320], 3, [20]
[321], 3, [21]
[0123], 0, [123]
[0132], 0, [132]
[0213], 0, [213]
[0231], 0, [231]
[0312], 0, [312]
[0321], 0, [321]
[1023], 1, [023]
[1032], 1, [032]
[1203], 1, [203]
[1230], 1, [230]
[1302], 1, [302]
[1320], 1, [320]
[2013], 2, [013]
[2031], 2, [031]
[2103], 2, [103]
[2130], 2, [130]
[2301], 2, [301]
[2310], 2, [310]
[3012], 3, [012]
[3021], 3, [021]
[3102], 3, [102]
[3120], 3, [120]
[3201], 3, [201]
[3210], 3, [210]
@.
sort(L, L) #-
[], []
[0], [0]
[1], [1]
[2], [2]
[3], [3]
[01], [01]
[02], [02]
[03], [03]
[10], [01]
[12], [12]
[13], [13]
[20], [02]
[21], [12]
[23], [23]
[30], [03]
[31], [13]
[32], [23]
[012], [012]
[013], [013]
[021], [012]
[023], [023]
[031], [013]
[032], [023]
[102], [012]
[103], [013]
[120], [012]
[123], [123]
[130], [013]
[132], [123]
[201], [012]
[203], [023]
[210], [012]
[213], [123]
[230], [023]
[231], [123]
[301], [013]
[302], [023]
[310], [013]
[312], [123]
[320], [023]
[321], [123]
[0123], [0123]
[0132], [0123]
[0213], [0123]
[0231], [0123]
[0312], [0123]
[0321], [0123]
[1023], [0123]
[1032], [0123]
[1203], [0123]
[1230], [0123]
[1302], [0123]
[1320], [0123]
[2013], [0123]
[2031], [0123]
[2103], [0123]
[2130], [0123]
[2301], [0123]
[2310], [0123]
[3012], [0123]
[3021], [0123]
[3102], [0123]
[3120], [0123]
[3201], [0123]
[3210], [0123]
@.
*partition(E,L,L,L) ##--
0, [], [], []
1, [], [], []
2, [], [], []
3, [], [], []
1, [0], [0], []
2, [0], [0], []
3, [0], [0], []
0, [1], [], [1]
2, [1], [1], []
3, [1], [1], []
0, [2], [], [2]
1, [2], [], [2]
3, [2], [2], []
0, [3], [], [3]
1, [3], [], [3]
2, [3], [], [3]
2, [01], [01], []
3, [01], [01], []
1, [02], [0], [2]
3, [02], [02], []
1, [03], [0], [3]
2, [03], [0], [3]
2, [10], [10], []
3, [10], [10], []
0, [12], [], [12]
3, [12], [12], []
0, [13], [], [13]
2, [13], [1], [3]
1, [20], [0], [2]
3, [20], [20], []
0, [21], [], [21]
3, [21], [21], []
0, [23], [], [23]
1, [23], [], [23]
1, [30], [0], [3]
2, [30], [0], [3]
0, [31], [], [31]
2, [31], [1], [3]
0, [32], [], [32]
1, [32], [], [32]
3, [012], [012], []
2, [013], [01], [3]
3, [021], [021], []
1, [023], [0], [23]
2, [031], [01], [3]
1, [032], [0], [32]
3, [102], [102], []
2, [103], [10], [3]
3, [120], [120], []
0, [123], [], [123]
2, [130], [10], [3]
0, [132], [], [132]
3, [201], [201], []
1, [203], [0], [23]
3, [210], [210], []
0, [213], [], [213]
1, [230], [0], [23]
0, [231], [], [231]
2, [301], [01], [3]
1, [302], [0], [32]
2, [310], [10], [3]
0, [312], [], [312]
1, [320], [0], [32]
0, [321], [], [321]
@.
@//E*O*F Release2/Examples/qs44.d//
chmod u=rw,g=r,o=r Release2/Examples/qs44.d
 
echo x - Release2/Examples/sort.d
sed 's/^@//' > "Release2/Examples/sort.d" <<'@//E*O*F Release2/Examples/sort.d//'
E: 0,1,2,3.
*L: *[],[0],[1],[2],[3],[01],[02],[03],[10],[12],[13],[20],[21],[23],[30],[31],[32],[012],[013],[021],[023],[031],[032],[102],[103],[120],[123],[130],[132],[201],[203],[210],[213],[230],[231],[301],[302],[310],[312],[320],[321],[0123],[0132],[0213],[0231],[0312],[0321],[1023],[1032],[1203],[1230],[1302],[1320],[2013],[2031],[2103],[2130],[2301],[2310],[3012],[3021],[3102],[3120],[3201],[3210].

*components(L,E,L) #--/-##
[0], 0, []
[1], 1, []
[2], 2, []
[3], 3, []
[01], 0, [1]
[02], 0, [2]
[03], 0, [3]
[10], 1, [0]
[12], 1, [2]
[13], 1, [3]
[20], 2, [0]
[21], 2, [1]
[23], 2, [3]
[30], 3, [0]
[31], 3, [1]
[32], 3, [2]
[012], 0, [12]
[013], 0, [13]
[021], 0, [21]
[023], 0, [23]
[031], 0, [31]
[032], 0, [32]
[102], 1, [02]
[103], 1, [03]
[120], 1, [20]
[123], 1, [23]
[130], 1, [30]
[132], 1, [32]
[201], 2, [01]
[203], 2, [03]
[210], 2, [10]
[213], 2, [13]
[230], 2, [30]
[231], 2, [31]
[301], 3, [01]
[302], 3, [02]
[310], 3, [10]
[312], 3, [12]
[320], 3, [20]
[321], 3, [21]
[0123], 0, [123]
[0132], 0, [132]
[0213], 0, [213]
[0231], 0, [231]
[0312], 0, [312]
[0321], 0, [321]
[1023], 1, [023]
[1032], 1, [032]
[1203], 1, [203]
[1230], 1, [230]
[1302], 1, [302]
[1320], 1, [320]
[2013], 2, [013]
[2031], 2, [031]
[2103], 2, [103]
[2130], 2, [130]
[2301], 2, [301]
[2310], 2, [310]
[3012], 3, [012]
[3021], 3, [021]
[3102], 3, [102]
[3120], 3, [120]
[3201], 3, [201]
[3210], 3, [210]
@.
sort(L, L) #-
[], []
[0], [0]
[1], [1]
[2], [2]
[3], [3]
[01], [01]
[02], [02]
[03], [03]
[10], [01]
[12], [12]
[13], [13]
[20], [02]
[21], [12]
[23], [23]
[30], [03]
[31], [13]
[32], [23]
[012], [012]
[013], [013]
[021], [012]
[023], [023]
[031], [013]
[032], [023]
[102], [012]
[103], [013]
[120], [012]
[123], [123]
[130], [013]
[132], [123]
[201], [012]
[203], [023]
[210], [012]
[213], [123]
[230], [023]
[231], [123]
[301], [013]
[302], [023]
[310], [013]
[312], [123]
[320], [023]
[321], [123]
[0123], [0123]
[0132], [0123]
[0213], [0123]
[0231], [0123]
[0312], [0123]
[0321], [0123]
[1023], [0123]
[1032], [0123]
[1203], [0123]
[1230], [0123]
[1302], [0123]
[1320], [0123]
[2013], [0123]
[2031], [0123]
[2103], [0123]
[2130], [0123]
[2301], [0123]
[2310], [0123]
[3012], [0123]
[3021], [0123]
[3102], [0123]
[3120], [0123]
[3201], [0123]
[3210], [0123]
@.
*lt(E,E) ##
0, 1
0, 2
0, 3
1, 2
1, 3
2, 3
@.
@//E*O*F Release2/Examples/sort.d//
chmod u=r,g=r,o=r Release2/Examples/sort.d
 
echo x - Release2/Src/Makefile
sed 's/^@//' > "Release2/Src/Makefile" <<'@//E*O*F Release2/Src/Makefile//'
# Vanilla makefile for distribution
# You may need to set local c compiler options

CFLAGS = -O2
 
@.SUFFIXES: .o .c .l .ln

@.c.o:	Makefile defns.i extern.i
#	lint -c $<
	cc $(CFLAGS) -c $<

@.c.ln:
	lint -c $<

SRC =	global.c main.c input.c output.c state.c\
	literal.c evaluatelit.c search.c determinate.c order.c\
	join.c utility.c finddef.c interpret.c prune.c constants.c

OBJ =	global.o main.o input.o output.o state.o\
	literal.o evaluatelit.o search.o determinate.o order.o\
	join.o utility.o finddef.o interpret.o prune.o constants.o

LINT =	global.ln main.ln input.ln output.ln state.ln\
	literal.ln evaluatelit.ln search.ln determinate.ln order.ln\
	join.ln utility.ln finddef.ln interpret.ln prune.ln constants.ln


ffoil:   $(OBJ) Makefile
#	lint -x $(LINT) -lm >,nittygritty
	cc -o ffoil2 $(OBJ) -lm


ffoilgt:  $(SRC) defns.i Makefile
	cat defns.i $(SRC) | egrep -v 'defns.i|extern.i' >ffoilgt.c
	cc -mips3 -non_shared -O4 -o ffoil2 ffoilgt.c -lm
	rm ffoilgt.c


$(OBJ): defns.i extern.i
@//E*O*F Release2/Src/Makefile//
chmod u=rw,g=r,o=r Release2/Src/Makefile
 
echo x - Release2/Src/constants.c
sed 's/^@//' > "Release2/Src/constants.c" <<'@//E*O*F Release2/Src/constants.c//'
/******************************************************************************/
/*									      */
/*	Routines concerned with discovering a plausible ordering of the	      */
/*	constants of each type.  There are three phases:		      */
/*	  *  finding possible orderings on pairs of relation arguments	      */
/*	  *  finding a partial ordering that satisfies as many of these	      */
/*	     as possible						      */
/*	  *  selecting a constant ordering consistent with the partial	      */
/*	     ordering							      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"

Boolean		**Table=Nil;		/* partial order for target type */

int		TTN,			/* target type number */
		NC,			/* size of target type */
		*TTCollSeq,		/* collation sequence */
		NArgOrders = 0,		/* number of possible arg orders */
		MaxConsistent;		/* max consistent arg orders */

ArgOrder	*ArgOrderList = Nil;


	/*  Find ordering for constants of each type  */

void  OrderConstants()
/*    --------------  */
{
    int		i, j;
    Boolean	**PO;
    Tuple	*NLT;

    ForEach(TTN, 1, MaxType)
    {
	if ( Type[TTN]->FixedPolarity ) continue;

	NC = Type[TTN]->NValues;
	TTCollSeq = Type[TTN]->CollSeq;
	Verbose(3) printf("\nOrdering constants of type %s\n",Type[TTN]->Name);

	if ( ! Table ) Table = AllocatePartOrd(MaxConst);

	FindArgumentOrders();

	if ( NArgOrders == 0 )
	{
	    Type[TTN]->Ordered = false;
	    Verbose(3) printf("\t\tunordered\n");
	    continue;
	}
	else
	{
	    Type[TTN]->Ordered = true;
	}

	/*  Assemble in Table the partial order consistent with the maximum
	    number of arg orders  */

	MaxConsistent = 0;
	PO = AllocatePartOrd(NC);
	ClearPartOrd(PO);
	Verbose(3) printf("\tFinding maximal consistent set\n");

	FindConsistentSubset(0, 0, PO);
	FreePartOrd(PO, NC);

	/*  Sort constants on number of entries in partial order; resolve
	    ties in favour of the initial constant order  */

	NLT = Alloc(NC, Tuple);
	ForEach(i, 0, NC-1)
	{
	    NLT[i] = Alloc(2, Const);
	    NLT[i][0] = Type[TTN]->Value[i];
	    FP(NLT[i][1]) = CountEntries(i+1) + i / (float) NC;
	}
	Quicksort(NLT, 0, NC-1, 1);

	/*  Change collation sequence and print message  */

	Verbose(3) printf("\tFinal order:\n\t\t");
	ForEach(i, 0, NC-1)
	{
	    j = NLT[i][0];
	    Type[TTN]->CollSeq[j] = i+1;
	    Verbose(3) printf("%s ", ConstName[j]);
	    pfree(NLT[i]);
	}
	Verbose(3) putchar('\n');

	pfree(NLT);
    }

    if ( Table ) FreePartOrd(Table, MaxConst);

    ForEach(i, 0, NArgOrders-1)
    {
	pfree(ArgOrderList[i]);
    }
    pfree(ArgOrderList);
}


	/*  Find potential orderings between pairs of arguments of each
	    relation for type TTN  */


void  FindArgumentOrders()
/*    ------------------  */
{
    int		i;
    Relation	R;

    ForEach(i, 0, MaxRel)
    {
	R = RelnOrder[i];
	if ( Predefined(R) || R->Arity < 2 ) continue;

	ExamineArgumentPairs(R, true, R->Pos);

	if ( R->Neg )
	{
	    ExamineArgumentPairs(R, false, R->Neg);
	}
    }
}



	/*  Find potential orderings between pairs of arguments of R where
	    relevant to the type under investigation  */


void  ExamineArgumentPairs(Relation R, Boolean Sign, Tuple *TP)
/*    --------------------    */
{
    int	FirstArg, SecondArg;

    Verbose(3)
	printf("\tChecking arguments of %s%s\n", Sign ? "" : "~", R->Name);

    ForEach(FirstArg, 1, R->Arity-1)
    {
	if ( ! Compatible[R->Type[FirstArg]][TTN] ) continue;

	ForEach(SecondArg, FirstArg+1, R->Arity)
	{
	    if ( ! Compatible[R->Type[SecondArg]][TTN] ) continue;

	    Verbose(3) 
		printf("\t\targuments %d,%d ", FirstArg, SecondArg);

	    ClearPartOrd(Table);
	    if ( ConsistentClosure(Table, TP, FirstArg, SecondArg) )
	    {
		Verbose(3) printf("are consistent\n");
		AddArgOrder(R, Sign, FirstArg, SecondArg);
	    }
	    else
	    {
		Verbose(3) printf("are not consistent\n");
	    }
	}
    }
}



	/*  Investigate args A and B of a set of tuples TP.  See whether each
	    pair of constants is consistent with TP; if so, add and form
	    closure  */

Boolean  ConsistentClosure(Boolean **PO, Tuple *TP, Var A, Var B)
/*       -----------------  */
{
    Const	K, L;
    int		i, j;

    while( *TP )
    {
	K = (*TP)[A];
	L = (*TP)[B];
	TP++;

	if ( K == MISSING_DISC || K == OUT_OF_RANGE ||
	     L == MISSING_DISC || L == OUT_OF_RANGE ) continue;

	/*  Not consistent if either constant missing from type or if
	    current pair in conflict with existing table  */

	if ( (i = TTCollSeq[K]) == 0 || (j = TTCollSeq[L]) == 0  ||
	     ! AddPair(PO, i, j) )
	{
	    return false;
	}
    }

    return true;
}
	


	/*  Note partial order between A and B; add to table if not already
	    there and generate closure.  Return false if the table is no
	    longer consistent  */

Boolean  AddPair(Boolean **PO, int A, int B)
/*       -------  */
{
    int	i, j;

    if ( PO[A][B] ) return true;		/* already there */
    else
    if ( A == B || PO[B][A] ) return false;	/* not consistent */

    ForEach(i, 1, NC)
    {
	if ( i == A || PO[i][A] )
	{
	    ForEach(j, 1, NC)
	    {
		if ( j == B || PO[B][j] )
		{
		    if ( PO[j][i] ) return false;

		    PO[i][j] = true;
		}
	    }
	}
    }

    return true;
}



void  AddArgOrder(Relation R, Boolean Sign, int A1, int A2)
/*    -----------  */
{
    ArgOrder	AO;

    if ( NArgOrders % 100 == 0 )
    {
	ArgOrderList = Realloc(ArgOrderList, NArgOrders+100, ArgOrder);
    }

    ArgOrderList[NArgOrders++] = AO = Alloc(1, struct _arg_ord_rec);

    AO->Rel  = R;
    AO->Sign = Sign;
    AO->A1   = A1;
    AO->A2   = A2;
    AO->In   = 0;
}


	/*  Routines for constant partial order tables  */

Boolean  **AllocatePartOrd(int Size)
/*         ---------------   */
{
    Boolean	**PO;
    int		i;

    PO = Alloc(Size+1, Boolean *);

    ForEach(i, 1, Size)
    {
	PO[i] = Alloc(Size+1, Boolean);
    }

    return PO;
}



void  FreePartOrd(Boolean **PO, int Size)
/*    -----------   */
{
    int i;

    ForEach(i, 1, Size)
    {
	pfree(PO[i]);
    }

    pfree(PO);
}


void  ClearPartOrd(Boolean **PO)
/*    ------------  */
{
    int i;

    ForEach(i, 1, NC)
    {
	memset(PO[i], false, NC+1);
    }
}


void  CopyPartOrd(Boolean **To, Boolean **From)
/*    -----------  */
{
    int i;

    ForEach(i, 1, NC)
    {
	memcpy(To[i], From[i], (NC+1)*sizeof(Boolean));
    }
}



void  FindConsistentSubset(int Included, int TryNext, Boolean **PO)
/*    --------------------  */
{
    Boolean	**CopyPO;
    ArgOrder	AO;
    Tuple	*Entries;
    int		i;

    if ( Included > MaxConsistent )
    {
	/*  Note best consistent partial order so far  */

	CopyPartOrd(Table, PO);
	MaxConsistent = Included;

	Verbose(3)
	{
	    printf("\t\tbest so far");
	    ForEach(i, 0, NArgOrders-1)
	    {
		AO = ArgOrderList[i];
		if ( AO->In )
		{
		    printf(" %s%s:", AO->Sign ? "" : "~", AO->Rel->Name);
		    if ( AO->In == 1 )
		    {
			printf("%d>%d", AO->A1, AO->A2);
		    }
		    else
		    {
			printf("%d>%d", AO->A2, AO->A1);
		    }
		}
	    }
	    putchar('\n');
	}
    }

    if ( TryNext >= NArgOrders ||
	 Included + (NArgOrders - TryNext) <= MaxConsistent )
    {
	return;
    }

    AO = ArgOrderList[TryNext];
    Entries = AO->Sign ? AO->Rel->Pos : AO->Rel->Neg;
    CopyPO = AllocatePartOrd(NC);
    CopyPartOrd(CopyPO, PO);

    if ( ConsistentClosure(PO, Entries, AO->A1, AO->A2) )
    {
	AO->In = 1;
	FindConsistentSubset(Included+1, TryNext+1, PO);
    }

    /*  Do not have to try both polarities of first argument ordering  */

    if ( Included > 0 )
    {
	CopyPartOrd(PO, CopyPO);
	if ( ConsistentClosure(PO, Entries, AO->A2, AO->A1) )
	{
	    AO->In = -1;
	    FindConsistentSubset(Included+1, TryNext+1, PO);
	}
    }

    CopyPartOrd(PO, CopyPO);
    AO->In = 0;
    FindConsistentSubset(Included, TryNext+1, PO);

    FreePartOrd(CopyPO, NC);
}



int  CountEntries(int K)
/*   ------------  */
{
    int i, Sum=0;

    ForEach(i, 1, NC)
    {
	if ( Table[K][i] ) Sum++;
    }

    return Sum;
}
@//E*O*F Release2/Src/constants.c//
chmod u=rw,g=r,o=r Release2/Src/constants.c
 
echo x - Release2/Src/defns.i
sed 's/^@//' > "Release2/Src/defns.i" <<'@//E*O*F Release2/Src/defns.i//'
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>


/******************************************************************************/
/*									      */
/*	Type definitions, including synonyms for structure pointers	      */
/*									      */
/******************************************************************************/


typedef unsigned char Var;

typedef char	Boolean,
		Ordering;	/* valid instances are <,=,>,# */


				/* A tuple is stored as an array of int / fp.

				   * T[0] = tuple number
						+ POSMARK (pos tuples)
				   * T[i] = constant number, or fp value

				   NB: If continuous (fp) values are used,
				       type Const must be the same size as
				       float --if this is not the case, change
				       Const to long.

				   Relation tuple sets are indexed by 3D array.

				   * I[i][j][k] = no. of kth tuple with T[j] = i

				   The last entry is followed by FINISH  */


typedef int			Const,
				*Tuple;

typedef int			***Index;

typedef struct _rel_rec		*Relation;

typedef struct _type_rec	*TypeInfo;

typedef struct _state_rec	State;

typedef struct _lit_rec		*Literal,
				**Clause;

typedef struct _arg_ord_rec	*ArgOrder;

typedef struct _poss_lit_rec	*PossibleLiteral;

typedef struct _backup_rec	*Alternative;

typedef struct _var_rec		*VarInfo;


/******************************************************************************/
/*									      */
/*	Structure definitions						      */
/*									      */
/******************************************************************************/

	/* State */

	/*  A state represents information about the search for a new
	    clause, including the tuples that satisfy the clause and
	    various counts for the clause and the tuples  */

struct _state_rec
{
	int		MaxVar,		/* highest variable */
			NPos,		/* number of pos tuples */
			NTot,		/* number of all tuples */
			NOrigPos,	/* original pos tuples covered */
			NOrigTot;	/* original tuples covered */
	Tuple		*Tuples;	/* training set */
	float		BaseInfo;	/* information per pos tuple */
};

	/* Literal */

struct _lit_rec
{
	char		Sign;		/* 0=negated, 1=pos, 2=determinate */
	Relation	Rel;
	Var		*Args;
	int		WeakLits;	/* value up to this literal */
	Ordering	*ArgOrders;	/* recursive lits: =, <, >, # */
	float		Bits;		/* encoding length */
};

	/* Relation */

	/* Note:

	   Relations are represented by the (positive) tuples they contain;
	   if the closed world assumption is not in force, negative tuples
	   known not to be in the relation can be given explicitly.

	   A key specifies a sensible way that a relation can be accessed
	   by noting which arguments must have bound values.  There can
	   be any number of keys; if there are none, all possible ways
	   of accessing the relation are ok. */
	   
struct _rel_rec
{
	char		*Name;
	int		Arity,		/* number of arguments */
			NKeys,		/* number of keys (0=all ok) */
			*Key,		/* keys, each packed in an int */
			*Type;		/* types of arguments */
	TypeInfo	*TypeRef;	/* redundant pointers to types */
	Tuple		*Pos,		/* positive tuples */
			*Neg;		/* negative tuples or Nil (CWA) */
	Index		PosIndex,	/* index for positive tuples */
			NegIndex;	/* ditto for explicit negative tuples */
	int		PosDuplicates,	/* number of duplicate pos tuples*/
			NTrialArgs,	/* number of args to try (gain) */
			NTried;		/* number of them evaluated */
	Clause		*Def;		/* definition is array of clauses */
	Boolean		BinSym,		/* true for binary symmetric relns */
			PossibleTarget,
			**PosOrder,	/* argument order info for R() */
			**NegOrder,	/*    "       "    "    " ~R() */
			*ArgNotEq;	/* args that cannot be the same var
					   (indexed by ArgPair) */
	float		Bits;		/* current encoding cost */
};

	/* Type */

struct _type_rec
{	char		*Name;		/* type name */
	Boolean		Continuous,	/* continuous (non-discrete) */
			Ordered,	/* ordered discrete type */
			FixedPolarity;
	int		NValues,	/* number of discrete constants */
			NTheoryConsts;  /* number of theory constants */
	Const		*Value,		/* constants */
			*TheoryConst;	/* theory constants */
	int		*CollSeq;	/* CollSeq[k] = x if (global) constant
					   k is x'th const of this type */
};


	/* Possible argument order -- used in discovering constant orders */

struct _arg_ord_rec
{
	Relation	Rel;		/* relation */
	Boolean		Sign;		/* sign */
	int		A1, A2,		/* A1 < A2 or A1 > A2 */
			In;		/* 0, -1 (reverse) or +1 */
};


	/* Structures used in backing up search */

struct _poss_lit_rec
{
	Relation	Rel;
	Boolean		Sign;
	Var		*Args;
	float		Gain,
			Bits;
	int		WeakLits,
			NewSize,
			TotCov,
			PosCov;
};

struct _backup_rec
{
	float		Value;
	Clause		UpToHere;
};


	/* Variables */

struct _var_rec
{
	char		*Name;
	int		Type,
			Depth;
	TypeInfo	TypeRef;
}; 


#define  FP(X)			(*((float *)(&X)))

	/*  The first four relations are predefined comparisons with
	    aliased names defined here  */

#define  EQVAR			Reln[0]		/* var = var */
#define  EQCONST		Reln[1]		/* var = theory const */
#define  GTVAR			Reln[2]		/* var > var */
#define  GTCONST		Reln[3]		/* var > threshold */

#define  Predefined(R)		(R==EQVAR||R==EQCONST||R==GTVAR||R==GTCONST)
#define  HasConst(R)		(R==EQCONST||R==GTCONST)

#define  Verbose(x)		if (VERBOSITY >= x)

#define  LN2	 		0.693147
#define  Log2(x)		(log((float) x)/LN2)
#define  Log2e			1.44269
#define  Log2sqrt2Pi		1.32575
#define  LogComb(n,r)		(Log2Fact(n) - Log2Fact(r) - Log2Fact((n)-(r)))
#define  Except(n,e)		((n) ? (1.1*(Log2(n) + LogComb(n, e))) : 0.0)
#define  Encode(n)		Except(AllTuples, n)

#define  Nil			0
#define  false			0 
#define  true			1 
#define  Max(a,b)		( (a)>(b) ? a : b ) 
#define  Min(a,b)		( (a)<(b) ? a : b ) 

#define  ForEach(V,First,Last)	for(V=First;V<=Last;V++) 

#define  Positive(T)		((T)[Target->Arity] == FnValue[(T)[0]])
#define  Undetermined(T)	((T)[Target->Arity] == UNBOUND )

#define  FalseBit		01
#define  TrueBit		02
#define	 CountedTBit		04
#define	 CountedFBit		010
#define  ResetFlag(A,B)		Flags[A] &= (~(B))
#define  SetFlag(A,B)		Flags[A] |= B
#define  TestFlag(A,B)		(Flags[A] & B)
#define  ClearFlags		memset(Flags,0,StartDef.NTot)

#define  BestLitGain		(NPossible ? Possible[1]->Gain : 0.0)

#define  MonitorWeakLits(W)	if (W) NWeakLits++; else NWeakLits=0

#define  Plural(n)		((n) != 1 ? "s" : "")
#define  ReadToEOLN		while ( getchar() != '\n' )

#define  Alloc(N,T)		(T *) pmalloc((N)*sizeof(T))
#define  AllocZero(N,T)		(T *) pcalloc(N, sizeof(T))
#define  Realloc(V,N,T)		V = (T *) prealloc(V, (N)*sizeof(T))

#define  MissingValue(R,A,X)	(MissingVals && MissingVal(R,A,X))
#define  ungetchar(A)		ungetc(A, stdin)

#define	 ArgPair(A2,A1)		(((A2-1)*(A2-2))/2 + A1-1)

	/*  The following are used to pack and unpack parameters into
	    argument lists.  AV must be the address of an int or float  */

#define  SaveParam(A,AV)	memcpy(A,AV,sizeof(Const))
#define  GetParam(A,AV)		memcpy(AV,A,sizeof(Const))



/******************************************************************************/
/*									      */
/*	Various constants						      */
/*									      */
/******************************************************************************/


#define  FINISH	   10000000	/* large constant used as a terminator */

#define	 UNBOUND    0357357	/* odd marker used in interpret.c, join.c */

#define  MISSING_DISC     1	/* missing value "?" is the first constant */

#define  MISSING_FP 0.03125	/* arbitrary number used as the floating
				   point equivalent of MISSING_DISC - if
				   it clashes with a genuine data value, just
				   change this */

#define	 OUT_OF_RANGE	  2	/* denotes constant outside closed world */



/******************************************************************************/
/*									      */
/*	Synopsis of functions						      */
/*									      */
/******************************************************************************/


	/* main.c */

void	  main(int Argc, char *Argv[]);

	/* utility.c */

void	  *pmalloc(unsigned arg);
void	  *prealloc(void * arg1, unsigned arg2);
void	  *pcalloc(unsigned arg1, unsigned arg2);
void	  pfree(void *arg);
float	  CPUTime();

	  /* input.c */

Boolean	  ReadType();
void	  ReadTypes();
Tuple	  ReadTuple(Relation R);
Tuple	  *ReadTuples(Relation R);
Relation  ReadRelation();
void	  ReadRelations();
int	  FindType(char *N);
char	  *CopyString(char *s);
void	  Error();
void	  DuplicateTuplesCheck(Relation R);
int	  CountDuplicates(Tuple *T, int N, int Left, int Right);
Boolean	  SymmetryCheck(Relation R);
char	  ReadName(char *s);
Const	  FindConstant(char *N, Boolean MustBeThere);
int	  Number(Tuple *T);
void	  CheckTypeCompatibility();
Boolean	  CommonValue(int N1, Const *V1, int N2, Const *V2);
Index	  MakeIndex(Tuple *T, int N, Relation R);
void	  UnequalArgsCheck(Relation R);
Boolean	  NeverEqual(Tuple *T, Var F, Var S);

	  /* output.c */

void	  PrintTuple(Tuple C, int N, TypeInfo *TypeRef, Boolean ShowPosNeg);
void	  PrintTuples(Tuple *TT, int N);
void	  PrintSpecialLiteral(Relation R, Boolean RSign, Var *A);
void	  PrintComposedLiteral(Relation R, Boolean RSign, Var *A);
void	  PrintLiteral(Literal L);
void	  PrintClause(Relation R, Clause C, Boolean Cut);
void	  PrintSimplifiedClause(Relation R, Clause C, Boolean Cut);
void	  Substitute(char *Old, char *New);
void	  PrintDefinition(Relation R);

	  /* literal.c */

void	  ExploreArgs(Relation R, Boolean CountFlag);
Boolean	  AcceptableKey(Relation R, int Key);
Boolean   Repetitious(Relation R, Var *A);
Boolean   SameArgs(int N, Var *A1, int MV1, Var *A2, int MV2, int LN);
void	  ExploreEQVAR();
void	  ExploreEQCONST();
void	  ExploreGTVAR();
void	  ExploreGTCONST();
Boolean	  TryArgs(Relation R, int This, int HiVar, int FreeVars, int MaxDepth,
	  	    int Key, Boolean TryMostGeneral, Boolean RecOK);
int	  EstimatePossibleArgs(int TNo);

	  /* evaluatelit.c */

void	  EvaluateLiteral(Relation R, Var *A, float LitBits, Boolean *Prune);
void	  PrepareForScan(Relation R, Var *A);
float	  NegThresh(int P, int P1);
Boolean	  TerminateScan(Relation R, Var *A);
Boolean	  Satisfies(int RN, Const V, Const W, Tuple Case);
void	  CheckForPrune(Relation R, Var *A);
void	  CheckNewVars(Tuple Case);
float	  Worth(int N, int P, int T, int UV);
float	  Info(int P, int T);
Boolean	  MissingVal(Relation R, Var *A, Tuple T);
Boolean	  Unknown(Var V, Tuple T);
void	  FindThreshold(Var *A);
void	  PossibleCut(float C);
int	  MissingAndSort(Var V, int Fp, int Lp);
void	  Quicksort(Tuple *Vec, int Fp, int Lp, Var V);

	  /* join.c */

Boolean	  Join(Tuple *T, Index TIX, Var *A, Tuple C, int N, Boolean YesOrNo);

	  /* state.c */

void	  OriginalState(Relation R);
void	  NewState(Literal L, int NewSize);
void	  FormNewState(Relation R, Boolean RSign, Var *A, int NewSize);
void	  AcceptNewState(Relation R, Var *A, int NewSize);
void	  RecoverState(Clause C, Boolean MakeNewClause);
void	  CheckSize(int SoFar, int Extra, int *NewSize, Tuple **TSAddr);
Tuple	  InitialiseNewCase(Tuple Case);
Tuple	  Extend(Tuple Case, Tuple Binding, Var *A, int N);
void	  CheckOriginalCaseCover();
int	  PosDefinite();
void	  FreeTuples(Tuple *TT, Boolean TuplesToo);
double	  Log2Fact(int n);

	  /* determinate.c */

Boolean	  GoodDeterminateLiteral(Relation R, Var *A, float LitBits);
void	  ProcessDeterminateLiterals();
Boolean	  SameVar(Var A, Var B);
void	  ShiftVarsDown(int s);

	  /* search.c */

void	  ProposeLiteral(Relation R, Boolean TF, Var *A,
	  		   int Size, float LitBits, int OPos, int OTot,
	  		   float Gain, Boolean Weak);
Boolean	  Recover();
void	  Remember(Literal L, int OPos, int OTot);
Literal	  MakeLiteral(int i);
Literal	  SelectLiteral();
void	  FreeLiteral(Literal L);
void	  FreeClause(Clause C);

	  /* order.c */

void	  ExamineVariableRelationships();
Boolean	  RecursiveCallOK(Var *A);
void	  AddOrders(Literal L);
void	  NoteRecursiveLit(Literal L);

	  /* finddef.c */

void	  FindDefinition(Relation R);
Clause	  FindClause();
void	  ExamineLiterals();
void	  GrowNewClause();
Boolean	  AllLHSVars(Literal L);
Boolean	  AllDeterminate();
float	  CodingCost(Clause C);

	  /* prune.c */

void	  PruneNewClause();
void	  CheckVariables();
Boolean	  TheoryConstant(Const C, TypeInfo T);
Boolean	  ConstantVar(Var V, Const C);
Boolean	  IdenticalVars(Var V, Var W);
Boolean	  Known(Relation R, Var V, Var W);
void	  Insert(Var V, Relation R, Var A1, Const A2);
Boolean	  Contains(Var *A, int N, Var V);
Boolean	  QuickPrune(Clause C, Var MaxBound, Boolean ValueBound);
Boolean	  SatisfactoryNewClause(int Cover, int Errs);
void	  Cleanup(Clause C);
void	  RenameVariables(Clause C);
Boolean	  RedundantLiterals(int ErrsNow);
Boolean	  EssentialBinding(int LitNo);
void	  ReplaceVariable(Var Old, Var New);
void	  SiftClauses();
void	  SimplifyClauses();
int	  CountErrs(int LastMod, Boolean Change);
Boolean	  UnsoundRecursion(Clause C);
Var	  HighestVarInDefinition(Relation R);
void	  RestoreTypeRefs();

	  /* interpret.c */

Boolean	  CheckRHS(Clause C);
Boolean	  Interpret(Relation R, Tuple Case);
void	  InitialiseValues(Tuple Case, int N);

	  /* constants.c */

void	  OrderConstants();
void	  FindArgumentOrders();
void	  ExamineArgumentPairs(Relation R, Boolean Sign, Tuple *TP);
Boolean	  ConsistentClosure(Boolean **Table, Tuple *TP, Var A, Var B);
Boolean	  AddPair(Boolean **Table, int A, int B);
void	  AddArgOrder(Relation R, Boolean Sig, int A1, int A2);
Boolean	  **AllocatePartOrd(int Size);
void	  FreePartOrd(Boolean **PO, int Size);
void	  ClearPartOrd(Boolean **PO);
void	  FindConsistentSubset(int Included, int TryNext, Boolean **PO);
int	  CountEntries(int K);
@//E*O*F Release2/Src/defns.i//
chmod u=rw,g=r,o=r Release2/Src/defns.i
 
echo x - Release2/Src/determinate.c
sed 's/^@//' > "Release2/Src/determinate.c" <<'@//E*O*F Release2/Src/determinate.c//'
/******************************************************************************/
/*									      */
/*	Routines for processing determinate literals			      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"


Boolean  GoodDeterminateLiteral(Relation R, Var *A, float LitBits)
/*       ----------------------  */
{
    int		MaxSoFar, PreviousMax, This, i, j;
    Var		V;
    Literal	L;
    Boolean	SensibleBinding=false;

    /*  See whether this determinate literal's bound variables were all bound
	by determinate literals on the same relation  */

    ForEach(j, 1, R->Arity)
    {
	SensibleBinding |= ( A[j] <= Target->Arity );
    }

    MaxSoFar = Target->Arity;

    for ( i = 0 ; ! SensibleBinding && i < NLit ; i++ )
    {
	if ( ! NewClause[i]->Sign ) continue;

	PreviousMax = MaxSoFar;

	ForEach(j, 1, NewClause[i]->Rel->Arity)
	{
	    if ( (V = NewClause[i]->Args[j]) > MaxSoFar ) MaxSoFar = V;
	}

	if ( NewClause[i]->Rel != R || NewClause[i]->Sign != 2 )
	{
	    ForEach(j, 1, R->Arity)
	    {
		SensibleBinding |= ( A[j] <= MaxSoFar && A[j] > PreviousMax );
	    }
	}
    }

    if ( ! SensibleBinding )
    {
	Verbose(3)
	{
	    printf("\tall vars bound by determinate lits on same relation\n");
	}
	return false;
    }

    /*  Record this determinate literal  */

    This = NLit + NDeterminate;
    if ( This && This%100 == 0 )
    {
        Realloc(NewClause, This+100, Literal);
    }

    L = NewClause[This] = AllocZero(1, struct _lit_rec);

    L->Rel  = R;
    L->Sign = 2;
    L->Bits = LitBits;
    L->Args = Alloc(R->Arity+1, Var);
    memcpy(L->Args, A, (R->Arity+1)*sizeof(Var));

    NDeterminate++;

    return true;
}



void  ProcessDeterminateLiterals()
/*    --------------------------  */
{
    int		PrevMaxVar, PrevTot, i, j, l, m;
    Literal	L;
    Var		V;
    Boolean	Unique, Changed;

    PrevMaxVar = Current.MaxVar;
    PrevTot = Current.NOrigTot;

    Verbose(1) printf("\nDeterminate literals\n");

    ForEach(l, 1, NDeterminate)
    {
	L = NewClause[NLit++];

	Verbose(1)
	{
	    putchar('\t');
	    PrintLiteral(L);
	}

	Changed = PrevMaxVar != Current.MaxVar;

	/*  Rename free variables  */

	ForEach(i, 1, L->Rel->Arity)
	{
	    if ( L->Args[i] > PrevMaxVar )
	    {
		L->Args[i] += Current.MaxVar - PrevMaxVar;

		if ( L->Args[i] > MAXVARS )
		{
		    Verbose(1) printf("\t\tno more variables\n");
		    NLit--;
		    MonitorWeakLits(PrevTot == Current.NOrigTot);
		    return;
		}
	    }
	}

	if ( Changed )
	{
	    Verbose(1)
	    {
		printf(" ->");
		PrintLiteral(L);
	    }
	    Changed = false;
	}

	if ( L->Rel == Target ) AddOrders(L);

        FormNewState(L->Rel, true, L->Args, Current.NTot);

	/*  Verify that new variables introduced by this determinate literal
	    don't replicate new variables introduced by previous determinate
	    literals.  [Note: new variables checked against old variables
	    in EvaluateLiteral() ]  */

	for ( i = Current.MaxVar+1 ; i <= New.MaxVar ; )
	{
	    Unique = true;

	    for ( j = PrevMaxVar+1 ; Unique && j <= Current.MaxVar ; j++ )
	    {
		Unique = ! SameVar(i, j);
	    }

	    if ( Unique )
	    {
		i++;
	    }
	    else
	    {
		j--;

		Verbose(1)
		{
		    printf(" %s=%s", Variable[i]->Name, Variable[j]->Name);
		}

		ShiftVarsDown(i);

		ForEach(V, 1, L->Rel->Arity)
		{
		    if ( L->Args[V] == i ) L->Args[V] = j;
		    else
		    if ( L->Args[V] >  i ) L->Args[V]--;
		}

		Changed = true;
	    }
	}

	/*  If no variables remain, delete this literal  */

	if ( Current.MaxVar == New.MaxVar )
	{
	    Verbose(1) printf(" (no new vars)");
	
	    NLit--;
	    ForEach(m, 1, NDeterminate-l)
	    {
		NewClause[NLit+m-1] = NewClause[NLit+m];
	    }

	    FreeTuples(New.Tuples, true);
	}
	else
	{
	    /* This determinate Literal is being kept in the clause */

	    if ( Changed )
	    {
		Verbose(1)
		{
		    printf(" ->");
		    PrintLiteral(L);
		}
	    }

	    AcceptNewState(L->Rel, L->Args, Current.NTot);
	    NDetLits++;

	    if ( L->Rel == Target ) NoteRecursiveLit(L);
	}

	Verbose(1) putchar('\n');
    }

    MonitorWeakLits(PrevTot == Current.NOrigTot);
}



	/*  See whether variable a is always the same as variable b in
	    all positive or undetermined tuples of the new state  */

Boolean  SameVar(Var A, Var B)
/*       -------  */
{
    Tuple	*TSP, Case;

    for ( TSP = New.Tuples ; Case = *TSP++; )
    {
	if ( ( Positive(Case) || Undetermined(Case) ) && Case[A] != Case[B] )
	{
	    return false;
	}
    }

    /*  If same, delete any negative tuples where different  */

    for ( TSP = New.Tuples ; Case = *TSP; )
    {
	if ( ! Positive(Case) && Case[A] != Case[B] )
	{
	    *TSP = New.Tuples[New.NTot-1];
	    New.NTot--;
	    New.Tuples[New.NTot] = Nil;
	}
	else
	{
	    TSP++;
	}
    }

    return true;
}



void  ShiftVarsDown(int s)
/*    -------------  */
{
    Tuple	*TSP, Case;
    Var		V;

    New.MaxVar--;

    for ( TSP = New.Tuples ; Case = *TSP++ ; )
    {
	ForEach(V, s, New.MaxVar)
	{
	    Case[V] = Case[V+1];
	}
    }

    ForEach(V, s, New.MaxVar)
    {
	Variable[V]->Type = Variable[V+1]->Type;
        Variable[V]->TypeRef = Variable[V+1]->TypeRef;
    }
}
@//E*O*F Release2/Src/determinate.c//
chmod u=rw,g=r,o=r Release2/Src/determinate.c
 
echo x - Release2/Src/evaluatelit.c
sed 's/^@//' > "Release2/Src/evaluatelit.c" <<'@//E*O*F Release2/Src/evaluatelit.c//'
/******************************************************************************/
/*									      */
/*	This group of routines has responsibility for evaluating a proposed   */
/*	literal.  There are many aspects that are checked simultaneously:     */
/*									      */
/*	  *  whether the literal is determinate				      */
/*	  *  whether this literal would result in a completed clause	      */
/*	  *  whether this literal would produce too many tuples		      */
/*	  *  pruning, both of this literal and of any specialisations of it   */
/*	  *  new variables that duplicate existing variables		      */
/*		- for all tuples					      */
/*		- for pos tuples (determinate literals)			      */
/*	  *  all new variables being bound to constants on pos tuples	      */
/*	  *  gain computation (including weak literal sequence check)	      */
/*	  *  adjusting records of best literals so far			      */
/*									      */
/*	The principles underlying this routine are:			      */
/*									      */
/*	  *  No literal may introduce a variable that duplicates an existing  */
/*	     variable (since the same effect could be obtained with a more    */
/*	     specific literal).						      */
/*	  *  No determinate literal may introduce a new variable that is      */
/*	     the same as an existing variable on pos tuples (since the more   */
/*	     specific literal would also be determinate).		      */
/*	  *  Exploration of a literal can cease as soon as it is clear that   */
/*	     the literal will not be used.  In some cases, it is also possible*/
/*	     to exclude other literals subsumed by this.  Such pruning is     */
/*	     tied to the calculation of gain and determinacy, and would need  */
/*	     to be altered if these are changed.			      */
/*									      */
/*	The various counts have components with the following meanings:	      */
/*									      */
/*		Pos	pos tuples					      */
/*		Neg	neg tuples					      */
/*		Tot	all tuples					      */
/*									      */
/*		T	pertaining to the unnegated literal R(A)	      */
/*		F	pertaining to ~R(A)				      */
/*		M	missing value (so neither)			      */
/*									      */
/*		Now	number of tuples in current state		      */
/*		Orig	number of tuples in original state		      */
/*		New	referring to state if use R(A) or ~R(A)		      */
/*									      */
/*	When computing Pos values,					      */
/*									      */
/*	  * an unbound tuple counts as 1 (potential pos)		      */
/*	  * an Orig tuple is counted as 1 only if all bindings are correct    */
/*	  * the status of each Orig tuple is maintained via flags	      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"


int
	OrigTPos,		/* orig pos tuples with extensions, R(A) */
	OrigTTot,
	OrigFPos,		/* ditto, ~R(A) */
	OrigFTot,

	NowTPos,		/* current pos tuples with extensions, R(A) */
	NowTTot,
	NowFPos,		/* ditto, ~R(A); NewFPos is identical */
	NowFNeg,
	NowFTot,		/* calculated for brevity at end */
	NowMPos,		/* pos tuples with missing values */
	NowMTot,		/* pos tuples with missing values */

	NewTPos,		/* pos tuples in new state, R(A) */
	NewTTot,

	NNewVars,		/* number of unbound vars in R(A) */
	NConstVars,		/* number of constant pseudo-vars  */
	NDuplicateVars,		/* number of duplications so far */
	NArgs,			/* number of relation arguments */
	BestCover,		/* coverage of best saveable clause */
	MinSaveableCover,	/* min coverage if clause saveable */
	MaxFPos,
	PossibleCuts;

Var
	BindingVar,
	*OldVar,		/* old var possibly equal to a new var */
	*NewVarPosn,		/* argument number of new var */
	*ConstVarPosn;		/* new vars with unchanging values */

Tuple
	FirstBinding;		/* check for consts masquerading as new vars */

Boolean
	Allocate = true,
	Bound,			/* whether fn value bound after literal */
	NegatedLiteralOK,	/* ~R(A) permissible  */
	*DifferOnNegs,		/* corresponding pair not identical */
	*FirstOccurrence,	/* first time variable appears in args */
	PossibleT,		/* R(A) is a candidate */
	PossibleF,		/* ~R(A) is a candidate */
	Determinate;		/* R(A) is determinate */

float
	MinUsefulGain,		/* min gain to affect outcome */
	MinPos,			/* min pos tuples to achieve MinUsefulGain */
	BestAccuracy,		/* accuracy of best saveable clause */
	MaxFNeg,		/* max FNeg tuples to avoid pruning ~R(A) */
	NewClauseBits;		/* if add literal to clause  */



	/*  Examine literals R(A) and ~R(A) (if appropriate)  */

void  EvaluateLiteral(Relation R, Var *A, float LitBits, Boolean *Prune)
/*    ---------------  */
{
    Boolean	Abandoned=false, RealDuplicateVars=false, WeakT, WeakF,
		SavedSign;
    Var		V, W;
    int		i, j, Coverage, Size, Potential;
    float	PosGain, NegGain, Gain, CurrentRatio, Accuracy, Extra;

    Verbose(3)
    {
	putchar('\t');
	PrintComposedLiteral(R, true, A);
    }

    if ( Prune ) *Prune = false;

    Extra = LitBits - Log2(NLit+1.001-NDetLits);
    NewClauseBits = ClauseBits + Max(0, Extra);

    PrepareForScan(R, A);

    if ( R == GTCONST )
    {
	/*  Find best threshold and counts in one go  */

	FindThreshold(A);
	if ( PossibleCuts > 0 ) LitBits += Log2(PossibleCuts);
	Determinate = false;
	NDuplicateVars = NConstVars = 0;
    }
    else
    {
	Abandoned = TerminateScan(R, A);
    }

    if ( Abandoned &&
	 NewTTot <= MAXTUPLES	/* not because out of room for tuples */ &&
	 NNewVars		/* new vars, so potential for pruning */ &&
	 Current.NPos - (NowFPos+NowMPos) < MinPos  /* R(A) has insuff gain */ )
    {
	/*  Check for possible pruning of more specific literals  */

	if ( NegatedLiteralOK )
	{
	    /*  A more specific ~R(A) will match more tuples.  In order to
		prune, we must be sure that the current ~R(A) matches
		enough neg tuples to ensure that the gain is not interesting.
		Assume all pos tuples other than those known to have missing
		values would be matched by a more specific ~R(A)  */

	    MaxFPos = Current.NPos - NowMPos;
	    MaxFNeg = NegThresh(MaxFPos, MaxFPos);

	    CheckForPrune(R, A);

	    *Prune = NowFNeg > MaxFNeg;
	}
	else
	{
	    /*  Can prune only when a more specific literal than R(A)
		could not be saveable  */

	    *Prune = OrigTPos < MinSaveableCover;
	}
    }

    NowFTot = NowFPos + NowFNeg;
    PossibleT &= ( Bound ? NewTPos > 0 : NowTTot > 0 );
    PossibleF &= ( Bound ? NowFPos > 0 : NowFTot > 0 );

    Verbose(3)
    {
	printf("  %d[%d/%d]", NowTPos, NewTPos, NewTTot);

	if ( NegatedLiteralOK || Abandoned )
	{
	    printf(" [%d/%d]", NowFPos, NowFTot);
	}
	printf("  ");
    }

    if ( Abandoned )
    {
	Verbose(3)
	{
	    printf(" abandoned");
	    if ( OutOfWorld ) printf(" ^");
	    printf("(%d%%)\n",
		   (100 * (NowTTot+NowFTot+NowMTot)) / Current.NTot);
	}

	return;
    }

    ForEach(i, 0, NDuplicateVars-1)
    {
	V = A[ NewVarPosn[i] ];
	W = OldVar[i];
	if ( W > Current.MaxVar ) W = A[ W-Current.MaxVar ];  /* special */
	if ( NewTPos || ! DifferOnNegs[i] )
	{
	    Verbose(3)
		printf(" %s%s%s",
		       Variable[W]->Name,
		       ( DifferOnNegs[i] ? "=+" : "=" ),
		       Variable[V]->Name);
	}

	RealDuplicateVars |= ! DifferOnNegs[i];
    }

    Verbose(3)
    {
	if ( NewTPos > 1 )
	{
	    ForEach(i, 0, NConstVars-1)
	    {
		V = A[ j=ConstVarPosn[i] ];

		printf(" %s=", Variable[V]->Name);

		if ( Variable[V]->TypeRef->Continuous )
		{
		    printf("%g", FP(FirstBinding[j]));
		}
		else
		{
		    printf("%s", ConstName[FirstBinding[j]]);
		}
	    }
	}
    }

    if ( Determinate )
    {
	/*  Check whether determinacy has been violated  */

	Determinate = ( Bound ? ! NDuplicateVars : ! RealDuplicateVars ) &&
		      NConstVars < NNewVars;

	Verbose(3) printf(" [%s]", Determinate ? "Det" : "XDet");
    }

    /*  Now assess gain if applicable.  Any literal that introduces a
	duplicate variable is unacceptable (the more specific literal
	would give the same result.  A literal R(A) that introduces a
	new variable that replicates an existing variable on pos tuples
	is also excluded -- the more specific literal would match the
	same number of pos tuples and perhaps fewer neg tuples  */

    if ( NewTPos ) PossibleT &= ! NDuplicateVars;

    if ( RealDuplicateVars || ( ! PossibleT && ! PossibleF ) )
    {
	Verbose(3) printf(" #\n");
	return;
    }

    /*  Due to arithmetic roundoff, ratios rather than gain are used to
	detect weak literals  */

    CurrentRatio = ( Current.NPos ?
		     Current.NPos/(float) Current.NTot :
		     1.0 / FnRange );

    if ( PossibleT )
    {
	if ( OrigTPos == OrigTTot && OrigTPos == StartClause.NTot )
	{
	    PosGain = MaxPossibleGain;
	}
	else
	{
	    Potential = ( ! Bound && ! BindingVar ? NowTTot : NowTPos );
	    PosGain = Worth(Potential, NewTPos, NewTTot, NNewVars-NConstVars);
	}
	WeakT = Bound && ! BindingVar && NowFTot <= 0 ||
		NewTPos/(NewTTot+1E-3) <= 0.9999 * CurrentRatio ||
		NewTTot > Current.NTot && 
		  OrigTPos == Current.NOrigPos && OrigTTot == Current.NOrigTot;
    }
    else
    {
	PosGain = 0.0;
    }

    if ( PossibleF )
    {
	if ( OrigFPos == OrigFTot && OrigFPos == StartClause.NPos )
	{
	    NegGain = MaxPossibleGain;
	}
	else
	{
	    Potential = ( ! Bound && ! BindingVar ? NowFTot : NowFPos );
	    NegGain = Worth(Potential, NowFPos, NowFTot, 0);
	}
	WeakF = Bound && ! BindingVar && NowTTot <= 0 ||
	        NowFPos/(NowFTot+1E-3) <= 0.9999 * CurrentRatio ||
		NowFTot > Current.NTot && 
		  OrigFPos == Current.NOrigPos && OrigFTot == Current.NOrigTot;
    }
    else
    {
	NegGain = 0.0;
    }

    Verbose(3)
    {
	printf(" gain %.1f", PosGain);
	if ( NegatedLiteralOK ) printf(",%.1f", NegGain);
    }

    /*  Weak literal sequence check  */

    if ( NWeakLits >= MAXWEAKLITS && ! Determinate &&
	 ( ! PossibleT || WeakT ) && ( ! PossibleF || WeakF ) )
    {
	Verbose(3) printf(" (weak)\n");
	return;
    }

    Verbose(3) putchar('\n');

    /*  Would the addition of this literal to the clause create the best
	saved clause so far? */

    if ( PossibleT &&
	 ( ! PossibleF ||
	   NewTPos / (float) NewTTot > NowFPos / (float) NowFTot ) )
    {
	SavedSign = 1;
	Accuracy = OrigTPos / (float) OrigTTot;
	Coverage = OrigTPos;
    }
    else
    if ( PossibleF )
    {
	SavedSign = 0;
	Accuracy = OrigFPos / (float) OrigFTot;
	Coverage = OrigFPos;
    }
    else
    {
	Accuracy = 0;
    }

    if ( Accuracy >= MINACCURACY-1E-3 &&
	 Coverage > 0 &&
	 ( Accuracy > BestAccuracy ||
	   Accuracy == BestAccuracy && Coverage > BestCover ) )
    {
	if ( ! AlterSavedClause )
	{
	    AlterSavedClause = AllocZero(1, struct _poss_lit_rec);
	}

	AlterSavedClause->Rel      = R;
	AlterSavedClause->Sign     = SavedSign;
	AlterSavedClause->Bits     = LitBits;
	AlterSavedClause->WeakLits = 0;
	AlterSavedClause->TotCov   = ( SavedSign ? OrigTTot : OrigFTot );
	AlterSavedClause->PosCov   = ( SavedSign ? OrigTPos : OrigFPos );

	Size = R->Arity+1;
	if ( HasConst(R) ) Size += sizeof(Const) / sizeof(Var);

	AlterSavedClause->Args = Alloc(Size, Var);
	memcpy(AlterSavedClause->Args, A, Size*sizeof(Var));
    }

    /*  Compute gains and save if appropriate  */

    Gain = Max(PosGain, NegGain);

    if ( Determinate &&
	 Gain < DETERMINATE * MaxPossibleGain &&
	 GoodDeterminateLiteral(R, A, LitBits) )
    {
	return;
    }

    if ( PosGain > 1E-3 &&
	 ( ! SavedClause || SavedClauseAccuracy < .999 || ! WeakT ) )
    {
	ProposeLiteral(R, true, A,
		       NowTTot, LitBits, OrigTPos, OrigTTot, 
		       PosGain, WeakT);
    }

    if ( NegGain > 1E-3 &&
	 ( ! SavedClause || SavedClauseAccuracy < .999 || ! WeakF ) )
    {
	ProposeLiteral(R, false, A,
		       NowFTot, LitBits, OrigFPos, OrigFTot, 
		       NegGain, WeakF);
    }
}



	/*  Initialise variables for scan.  NOTE: this assumes that
	    Current.BaseInfo and MaxPossibleGain have been set externally.  */


void  PrepareForScan(Relation R, Var *A)
/*    --------------  */
{
    int i, j, PossibleVarComps;
    Var V, W, VP;

    /*  First time through, allocate arrays  */

    if ( Allocate )
    {
	Allocate = false;

	PossibleVarComps = MAXARGS * MAXVARS + (MAXARGS * (MAXARGS-1) / 2);
	OldVar       = Alloc(PossibleVarComps, Var);
	NewVarPosn   = Alloc(PossibleVarComps, Var);

	ConstVarPosn    = Alloc(MAXARGS+1, Var);
	DifferOnNegs    = Alloc((MAXARGS+1)*(MAXVARS+1), Boolean);
	FirstOccurrence = Alloc(MAXVARS+MAXARGS, Boolean);
	FirstBinding    = Alloc(MAXARGS+1, Const);
    }

    OrigTPos = 0;
    OrigTTot = 0;
    OrigFPos = 0;
    OrigFTot = 0;

    NowTPos = 0;
    NowTTot = 0;
    NowFPos = 0;
    NowFNeg = 0;
    NowMPos = 0;
    NowMTot = 0;

    NewTPos = 0;
    NewTTot = 0;

    NDuplicateVars = NNewVars = NConstVars = 0;

    NArgs = R->Arity;
    memset(FirstOccurrence, true, Current.MaxVar+NArgs);

    OutOfWorld = false;
    Bound = ( ! Undetermined(Current.Tuples[0]) );
    BindingVar = 0;

    ForEach(i, 1, NArgs)
    {
	V = A[i];

	if ( ! Bound && V == Target->Arity )
	{
	    Bound = true;
	    BindingVar = i;
	}

	if ( V > Current.MaxVar && FirstOccurrence[V] )
	{
	    NNewVars++;
	    FirstOccurrence[V] = false;

	    ConstVarPosn[NConstVars++] = i;

	    ForEach(W, 1, Current.MaxVar)
	    {
		if ( ! Compatible[Variable[W]->Type][R->Type[i]] ||
		     Current.Tuples[0][W] == UNBOUND ) continue;

		NewVarPosn[NDuplicateVars]   = i;
		OldVar[NDuplicateVars]       = W;
		DifferOnNegs[NDuplicateVars] = false;
		NDuplicateVars++;
	    }
	}
    }

    /*  New variables shouldn't replicate each other, either  */

    ForEach(i, 0, NConstVars-2)
    {
	VP = ConstVarPosn[i];
	V  = A[VP];

	ForEach(j, i+1, NConstVars-1)
	{
	    W = ConstVarPosn[j];

	    if ( ! Compatible[Variable[V]->Type][R->Type[W]] ) continue;

	    NewVarPosn[NDuplicateVars]   = W;
	    OldVar[NDuplicateVars]       = Current.MaxVar+VP;	/* special */
	    DifferOnNegs[NDuplicateVars] = false;
	    NDuplicateVars++;
	}
    }

    ClearFlags;

    PossibleT = true;
    PossibleF = NegatedLiteralOK =
		  ( NEGLITERALS ||
		    R == GTVAR || R == GTCONST ||
		    ( NEGEQUALS && ( R == EQVAR || R == EQCONST ) ) );

    Determinate = NNewVars > 0;

    /*  The minimum gain that would be of interest is just enough to give
	a literal a chance to be saved by the backup procedure or, if
	there are determinate literals, to reach the required fraction
	of the maximum possible gain  */

    MinUsefulGain = NPossible < MAXPOSSLIT ? MINALTFRAC * BestLitGain :
		    Max(Possible[MAXPOSSLIT]->Gain, MINALTFRAC * BestLitGain);

    if ( NDeterminate && MinUsefulGain < DETERMINATE * MaxPossibleGain )
    {
	MinUsefulGain = DETERMINATE * MaxPossibleGain;
    }

    /*  Set thresholds for pos tuples  */

    MinPos = MinUsefulGain / Current.BaseInfo - 0.001;

    /*  Now check coverage required for a saveable clause that would pass
	the MDL criterion.  Don't worry about long saveable clauses.  */

    if ( AlterSavedClause )
    {
	BestCover    = AlterSavedClause->PosCov;
	BestAccuracy = BestCover / (float) AlterSavedClause->TotCov;
    }
    else
    {
	BestCover    = SavedClauseCover;
	BestAccuracy = SavedClauseAccuracy;
    }

    if ( NLit < 5 )
    {
	MinSaveableCover = BestCover+1;
	while ( Encode(MinSaveableCover) <= NewClauseBits ) MinSaveableCover++;
    }
    else
    {
	MinSaveableCover = StartDef.NPos;
    }
}



	/*  Make a pass through the tuples, terminating if it becomes clear
	    that neither R(A) or ~R(A) can achieve the minimum useful gain.
	    Since all pos tuples appear first in the tuple sets, thresholds
	    for NewTNeg and NowFNeg can be set when the first neg tuple
	    is encountered.  */


#define  TermTest(Cond, Test)\
	    if ( Cond && Test && ! Determinate ) {\
		 Cond = false; if ( ! PossibleT && ! PossibleF ) return true; }
		

Boolean  TerminateScan(Relation R, Var *A)
/*       -------------  */
{
    Tuple	*TSP, Case;
    Boolean	BuiltIn=false, CheckMDL, CheckTerm, PosCase;
    int		i, RN, MaxCover, NCorrect, OrigPos=0;
    Const	X2;
    float	NewTNegThresh, NowFNegThresh;

    if ( Predefined(R) )
    {
	BuiltIn = true;
	RN = (int) R->Pos;
	if ( HasConst(R) )
	{
	    GetParam(&A[2], &X2);
	}
	else
	{
	    X2 = A[2];
	}
    }

    CheckMDL = CheckTerm = Bound && ! BindingVar;

    for ( TSP = Current.Tuples ; Case = *TSP++ ; )
    {
	if ( CheckMDL && ! Positive(Case) )
	{
	    /*  Encoding length checks  */

	    PossibleT &= Encode(OrigTPos) > NewClauseBits;
	    PossibleF &= Encode(OrigFPos) > NewClauseBits;

	    if ( ! PossibleT && ! PossibleF )
	    {
		Verbose(3)
		{
		    printf("  MDL prune %d,%d", OrigTPos, OrigFPos);
		}
		return true;
	    }

	    /*  Set thresholds now that NowTPos and NowFPos are known  */

	    NewTNegThresh = ( NNewVars && BestLitGain < 1E-2 ? MAXTUPLES :
			      NegThresh(NowTPos, NewTPos) );
	    NowFNegThresh = NegThresh(NowFPos, NowFPos);
	    CheckMDL = false;
	}
		
	if ( MissingValue(R, A, Case) )
	{
	    NowMTot++;
	    if ( Positive(Case) ) NowMPos++;
	    NFound = 0;
	}
	else
	if ( BuiltIn ? Satisfies(RN, A[1], X2, Case) :
	     Join(R->Pos, R->PosIndex, A, Case, NArgs, false) )
	{
	    /*  R(A) is barred if it would introduce an out-of-world constant.
		Note: can't use TermTest() since check for Determinate does
		not matter  */

	    if ( OutOfWorld )
	    {
		PossibleT = false;
		if ( ! PossibleF ) return true;
	    }
		
	    /*  Extensions of this tuple from R(A)  */

	    CheckNewVars(Case);

	    NowTTot++;
	    NewTTot += NFound;

	    TermTest(PossibleT,
		     NewTTot > MAXTUPLES);

	    /*  Is this a positive tuple?  */

	    if ( BindingVar )
	    {
		NCorrect = 0;
		ForEach(i, 0, NFound-1)
		{
		    if ( ! BuiltIn ) BoundValue = Found[i][BindingVar];

		    if ( BoundValue == FnValue[Case[0]] )
		    {
			NCorrect++;
		    }
		}
		PosCase = (NCorrect == NFound);
	    }
	    else
	    {
		PosCase = Positive(Case);
	    }

	    /*  Adjust OrigT counts.  Note: Want to increment OrigTPos
		only if *all* extensions of this tuple have the correct
		function value  */

	    if ( ! TestFlag(Case[0], TrueBit) )
	    {
		SetFlag(Case[0], TrueBit);
		OrigTTot++;
		if ( PosCase )
		{
		    OrigTPos++;
		    SetFlag(Case[0], CountedTBit);
		    if ( ! TestFlag(Case[0], FalseBit) ) OrigPos++;
		}
	    }
	    else
	    if ( ! PosCase && TestFlag(Case[0], CountedTBit) )
	    {
		ResetFlag(Case[0], CountedTBit);
		OrigTPos--;
	    }

	    if ( PosCase )
	    {
		NowTPos++;
		NewTPos += NFound;

		/*  If all remaining pos tuples go to NowFPos, are there
		    sufficient to make ~R(A) viable?
		    Note: do not abandon ~R(A) if it could lead to a
		    saveable clause (assuming all remaining original
		    pos tuples are covered by ~R(A))  */

		MaxCover = OrigFPos + (Current.NOrigPos - OrigPos);
		TermTest(PossibleF,
			 CheckTerm && MaxCover < MinSaveableCover &&
			 Current.NPos - (NowTPos + NowMPos) < MinPos-1E-3);
	    }
	    else
	    {
		/*  We already know the final number of NewTPos tuples.
		    Are there now enough NewTNeg tuples to make the gain of
		    R(A) insufficient?  (Note: R(A) matches a negative
		    tuple so saveability is not an issue.)  */

		TermTest(PossibleT,
			 CheckTerm &&
			 (NewTTot - NewTPos) > NewTNegThresh+1E-3);
	    }
	}
	else
	{
	    PosCase = Positive(Case);

	    /*  Adjust OrigF counts (see note on OrigT counts)  */

	    if ( ! TestFlag(Case[0], FalseBit) )
	    {
		SetFlag(Case[0], FalseBit);
		OrigFTot++;
		if ( Positive(Case) )
		{
		    OrigFPos++;
		    SetFlag(Case[0], CountedFBit);
		    if ( ! TestFlag(Case[0], TrueBit) ) OrigPos++;
		}
	    }
	    else
	    if ( ! PosCase && TestFlag(Case[0], CountedFBit) )
	    {
		ResetFlag(Case[0], CountedFBit);
		OrigFPos--;
	    }

	    if ( PosCase )
	    {
		NowFPos++;

		/*  If all remaining pos tuples go to NowTPos, are there
		    sufficient to make R(A) viable?
		    Note: don't kill R(A) if it could lead to a saveable
		    clause when all remaining original pos tuples covered
		    by R(A)  */

		MaxCover = OrigTPos + (Current.NOrigPos - OrigPos);
		TermTest(PossibleT,
			 CheckTerm && MaxCover < MinSaveableCover &&
			 Current.NPos - (NowFPos + NowMPos) < MinPos-1E-3);
	    }
	    else
	    {
		NowFNeg++;

		/*  We already know the final number of NowFPos tuples.
		    Are there already enough NowFNeg tuples to make the gain of
		    ~R(A) insufficient?  (Saveability as above.)  */

		TermTest(PossibleF,
			 CheckTerm &&
			 NowFNeg > NowFNegThresh+1E-3);
	    }
	}

	Determinate &= ( PosCase || ! CheckTerm ? NFound == 1 : NFound <= 1 );
    }

    return false;
}



	/*  If there are unbound variables, try to satisfy the
	    pruning criterion for more specific literals.

	    A more specific negated literal will cover more tuples;
	    NowFNeg must be great enough so that, if all positive tuples
	    were covered by ~R(A), the gain would still be too low.  */


void  CheckForPrune(Relation R, Var *A)
/*    -------------  */
{
    Tuple	*TSP, Case;
    int		RemainingNeg;

    RemainingNeg = (Current.NTot - Current.NPos)
		   - (NowMTot - NowMPos) - (NowTTot - NowTPos) - NowFNeg;

    for ( TSP = Current.Tuples + (NowMTot+NowTTot+NowFPos+NowFNeg) ;
	  Case = *TSP++ ; )
    {
	if ( Positive(Case) || MissingValue(R, A, Case) )
	{	
	    continue;
	}

	if ( ! Join(R->Pos, R->PosIndex, A, Case, NArgs, true) )
	{
	    NowFNeg++;

	    /*  See if have found enough  */

	    if ( NowFNeg > MaxFNeg ) break;
	}

	RemainingNeg--;

	/*  See whether not enough left  */

	if ( NowFNeg + RemainingNeg <= MaxFNeg ) break;
    }
}



	/*  Check new variables for non-utility, specifically
	    *  replicating existing variables on all or pos tuples
	    *  all being bound to constants on pos tuples  */


void  CheckNewVars(Tuple Case)
/*    ------------  */
{
    Var		P;
    Const	OldVarVal;
    int		i, j, Col;
    Boolean	Remove;

    for ( i = 0 ; i < NDuplicateVars ; i++ )
    {
	P = OldVar[i];
	if ( P <= Current.MaxVar ) OldVarVal = Case[P];

	Col = NewVarPosn[i];

 	Remove = false;
	for ( j = 0 ; j < NFound && ! Remove ; j++ )
	{
	    if ( Found[j][Col] != ( P <= Current.MaxVar ? OldVarVal :
				                  Found[j][P-Current.MaxVar] ) )
	    {
		if ( Positive(Case) ||
		BindingVar && Found[j][BindingVar] == FnValue[Case[0]] )
		{
		    Remove = true;
		}
		else
		{
		    DifferOnNegs[i] = true;
		}
	    }
	}

	if ( Remove )
	{
	    NDuplicateVars--;

	    for ( j = i ; j < NDuplicateVars ; j++ )
	    {
		NewVarPosn[j]   = NewVarPosn[j+1];
		OldVar[j]       = OldVar[j+1];
		DifferOnNegs[j] = DifferOnNegs[j+1];
	    }
	    i--;
	}
    }

    /*  Check for new vars bound to constants  */

    if ( NConstVars )
    {
	if ( ! NowTTot )
	{
	    memcpy(FirstBinding, Found[0], (NArgs+1)*sizeof(Const));
	}

	ForEach(i, 0, NConstVars-1)
	{
	    Col = ConstVarPosn[i];

	    for ( j = 0 ; j < NFound ; j++ )
	    {
		if ( FirstBinding[Col] == Found[j][Col] ) continue;

		NConstVars--;

		for ( j = i ; j < NConstVars ; j++ )
		{
		    ConstVarPosn[j] = ConstVarPosn[j+1];
		}

		i--;
		break;
	    }
	}
    }
}



	/*  Compute the maximum number N1 of neg tuples that would allow
	    P1 pos tuples (P orig pos tuples) to give a gain >= threshold.
	    The underlying relation is
		P * (Current.BaseInfo + log(P1/(P1+N1)) >= MinUsefulGain
	    where N1 is adjusted by the sampling factor.

	    NOTE: This is the inverse of the gain calculation in Worth.
	    If one is changed, the other must be modified accordingly  */


float  NegThresh(int P, int P1)
/*     ---------  */
{
    return ( ! Bound && ! BindingVar ? MAXTUPLES :
	     P <= 0 ? 0.0 :
	     (P1+1) * (exp(LN2 * (Current.BaseInfo - MinUsefulGain/P)) - 1) );
}



	/*  Compute aggregate gain from a test on relation R, tuple T.
	    The Basic gain is the number of positive tuples * information
	    gained regarding each; but there is a minor adjustment:
	      - a literal that has some positive tuples and no gain but
		introduces one or more new variables, is given a slight gain  */


float  Worth(int N, int P, int T, int UV)
/*     -----  */
{
    float G, TG;

    if ( ! N ) return 0.0;

    if ( ! P && UV )
    {
	return 1E-3 + 1E-3 * N / (T + 1.0);
    }

    TG = N * (G = Current.BaseInfo - Info(P, T));

    if ( G < 1E-3 && N && UV )
    {
	return 1E-3 + UV * 0.0001;  /* very small notional gain */
    }
    else
    {
	return TG;
    }
}



	/*  The ratio P/T is tweaked slightly to (P+1)/(T+1) so that, if
	    two sets of tuples have the same proportion of pos tuples,
	    the smaller is preferred.  The reasoning is that it is easier
	    to filter out all neg tuples from a smaller set.  If you don't
	    like this idea and change it back to P/T, NegThresh must be
	    changed also  */


float  Info(int P, int T)
/*     ----  */
{
    return ( P ? Log2(T+1) - Log2(P+1) :
		 Log2(FnRange) );
}



Boolean  MissingVal(Relation R, Var *A, Tuple T)
/*       ----------       */
{
    register Var i, V;

    ForEach(i, 1, R->Arity)
    {
	V = A[i];

        if ( V <= Current.MaxVar && Unknown(V, T) ) return true;
    }

    return false;
}



Boolean  Unknown(Var V, Tuple T)
/*       -------  */
{
    return ( Variable[V]->TypeRef->Continuous ?
	     FP(T[V]) == MISSING_FP : T[V] == MISSING_DISC );
}



	/*  See whether a case satisfies built-in relation RN  */

Boolean  Satisfies(int RN, Const V, Const W, Tuple Case)
/*       ---------  */
{
    switch ( RN )
    {
	case 0:	/* EQVAR */
		NFound = ( Case[V] == Case[W] ||
			   Case[V] == UNBOUND && (BoundValue = Case[W]) ||
			   Case[W] == UNBOUND && (BoundValue = Case[V]) );
		break;

	case 1:	/* EQCONST */
		NFound = ( Case[V] == W ||
			   Case[V] == UNBOUND && (BoundValue = W) );
		break;

	case 2:	/* GTVAR */
		NFound = ( Case[V] != UNBOUND && FP(Case[V]) > FP(Case[W]) );
		break;

	case 3:	/* GTCONST */
		NFound = ( Case[V] != UNBOUND && FP(Case[V]) > FP(W) );
		break;

	default:	exit(0);
    }

    return NFound;
}



	/*  The following stuff calculates thresholds for GTCONST relations.

	    The pos and neg tuples are sorted separately, then merged.
	    The current value is acceptable as a threshold unless it is
	    in the middle of a run of tuples of the same sign or a run
	    of the same value  */


float	BestGain, BestThresh;
int	BestTPos, BestTTot;


void  FindThreshold(Var *A)
/*    -------------  */
{
    Tuple	*ScanPos, *ScanNeg, Case;
    float	PosVal, NegVal, ThisVal, PrevVal=(-1E30);
    Var		V;
    int		ThisSign, NextSign, Signs=0, i;

    V = A[1];

    BestGain = -1E10;
    PossibleCuts = 0;

    NowMPos = MissingAndSort(V, 0, Current.NPos-1);
    NowMTot = NowMPos + MissingAndSort(V, Current.NPos, Current.NTot-1);

    NowTPos = Current.NPos - NowMPos;
    NowTTot = Current.NTot - NowMTot;

    ScanPos = &Current.Tuples[NowMPos];
    ScanNeg = &Current.Tuples[Current.NPos + (NowMTot - NowMPos)];
    PosVal = ( NowTPos ? FP(((*ScanPos)[V])) : 1E30 );
    NegVal = ( NowTTot > NowTPos ? FP(((*ScanNeg)[V])) : 1E30 );

    while ( NowTTot >= 1 )
    {
	if ( PosVal <= NegVal )
	{
	    NowTPos--;
	    NowTTot--;
	    ScanPos++;
	    ThisVal = PosVal;
	    PosVal = ( NowTPos ? FP(((*ScanPos)[V])) : 1E30 );
	    ThisSign = 1;
	}
	else
	{
	    NowTTot--;
	    ScanNeg++;
	    ThisVal = NegVal;
	    NegVal = ( NowTTot > NowTPos ? FP(((*ScanNeg)[V])) : 1E30 );
	    ThisSign = 2;
	}

	if ( ThisVal == PrevVal )
	{
	    Signs |= ThisSign;
	}
	else
	{
	    Signs = ThisSign;
	    PrevVal = ThisVal;
	    PossibleCuts++;
	}

	NextSign = ( PosVal == NegVal ? 3 : PosVal < NegVal ? 1 : 2 );

	if ( NowTTot && ThisVal != PosVal && ThisVal != NegVal &&
	     (Signs | NextSign) == 3 )
	{
	    PossibleCut(ThisVal);
	}
    }
    PossibleCuts--;

    /*  Fix up all required counts  */

    if ( BestGain >= 0 )
    {
	NewTPos = NowTPos = BestTPos;
	NewTTot = NowTTot = BestTTot;

	NowFPos = Current.NPos - NowTPos - NowMPos;
	NowFNeg = (Current.NTot - Current.NPos)
		    - (NowTTot - NowTPos) - (NowMTot - NowMPos);

	SaveParam(&A[2], &BestThresh);
	Verbose(3) printf("%g", BestThresh);

	/* Now determine coverage of original tuples  */

	ClearFlags;
	OrigTPos = OrigTTot = OrigFPos = OrigFTot = 0;

	ForEach(i, 0, Current.NTot-1)
	{
	    Case = Current.Tuples[i];
	    ThisVal = FP(Case[V]);

	    if ( ThisVal == MISSING_FP ) continue;

	    if ( ThisVal > BestThresh )
	    {
		if ( ! TestFlag(Case[0], TrueBit) )
		{
		    SetFlag(Case[0], TrueBit);
		    OrigTTot++;
		    if ( Positive(Case) ) OrigTPos++;
		}
	    }
	    else
	    {
		if ( ! TestFlag(Case[0], FalseBit) )
		{
		    SetFlag(Case[0], FalseBit);
		    OrigFTot++;
		    if ( Positive(Case) ) OrigFPos++;
		}
	    }
	}

	/*  Encoding length checks  */

	PossibleT &= Encode(OrigTPos) > NewClauseBits;
	PossibleF &= Encode(OrigFPos) > NewClauseBits;
    }
    else
    {
	NewTPos = NowTPos = Current.NPos;
	NewTTot = NowTTot = Current.NTot;
	NowFPos = NowFNeg = 0;
	PossibleT = PossibleF = false;
    }
}



void  PossibleCut(float C)
/*    -----------  */
{
    float	TGain, FGain, Better;

    TGain = Worth(NowTPos, NowTPos, NowTTot, 0);
    FGain = Worth(Current.NPos-NowTPos-NowMPos,
		  Current.NPos-NowTPos-NowMPos,
		  Current.NTot-NowTTot-NowMTot, 0);
    Better = Max(TGain, FGain);

    if ( Better > BestGain )
    {
	BestGain   = Better;
	BestThresh = C;
	BestTPos   = NowTPos;
	BestTTot   = NowTTot;
    }
}



	/*  Count missing values and sort the remainder  */

Tuple			Hold;
#define  Swap(V,A,B)	{ Hold = V[A]; V[A] = V[B]; V[B] = Hold; }


int  MissingAndSort(Var V, int Fp, int Lp)
/*   --------------  */
{
    int	i, Xp;

    /*  Omit and count unknown values */

    Xp = Fp;
    ForEach(i, Fp, Lp)
    {
	if ( FP(Current.Tuples[i][V]) == MISSING_FP )
	{
	    Swap(Current.Tuples, Xp, i);
	    Xp++;
	}
    }

    Quicksort(Current.Tuples, Xp, Lp, V);

    return Xp - Fp;
}



void  Quicksort(Tuple *Vec, int Fp, int Lp, Var V)
/*    ---------  */
{
    register int Middle, i;
    register float Thresh;

    if ( Fp < Lp )
    {
	Thresh = FP(Vec[ (Fp+Lp)/2 ][V]);

	/*  Isolate all items with values < threshold  */

	Middle = Fp;

	for ( i = Fp ; i <= Lp ; i++ )
	{ 
	    if ( FP(Vec[i][V]) < Thresh )
	    { 
		if ( i != Middle ) Swap(Vec, Middle, i);
		Middle++; 
	    } 
	} 

	/*  Sort the lower values  */

	Quicksort(Vec, Fp, Middle-1, V);

	/*  Extract all values equal to the threshold  */

	for ( i = Middle ; i <= Lp ; i++ )
	{
	    if ( FP(Vec[i][V]) == Thresh )
	    { 
		if ( i != Middle ) Swap(Vec, Middle, i);
		Middle++;
	    } 
	} 

	/*  Sort the higher values  */

	Quicksort(Vec, Middle, Lp, V);
    }
}
@//E*O*F Release2/Src/evaluatelit.c//
chmod u=rw,g=r,o=r Release2/Src/evaluatelit.c
 
echo x - Release2/Src/extern.i
sed 's/^@//' > "Release2/Src/extern.i" <<'@//E*O*F Release2/Src/extern.i//'
/******************************************************************************/
/*									      */
/*	Variables defined in global.c					      */
/*									      */
/******************************************************************************/


extern Boolean
	NEGLITERALS,
	NEGEQUALS,
	UNIFORMCODING,
	SIMPLIFY,

	MissingVals,
	AnyPartialOrder,
	*Barred,
	*VarUsed,
	OutOfWorld;


extern float
	MINACCURACY,
	MINALTFRAC,
	DETERMINATE;

extern int
	MAXVARS,
	MAXARGS,
	MAXWEAKLITS,

	MAXPOSSLIT,
	MAXALTS,
	MAXVARDEPTH,
	MAXRECOVERS,
	MAXTUPLES,
	VERBOSITY,

	MaxConst,
	MaxType,
	MaxRel,

	FnRange,
	AllTuples,
	NCl,
	NLit,
	NDetLits,
	NWeakLits,
	SavedClauseCover;


extern PossibleLiteral
	AlterSavedClause;


extern char
	**ConstName,
	*Flags;


extern Relation
	*Reln,
	*RelnOrder,
	Target;


extern State
	StartDef,
	StartClause,
	Current,
	New;


extern float
	*LogFact,
	MaxPossibleGain,
	ClauseBits,
	AvailableBits,
	SavedClauseAccuracy;


extern Clause
	NewClause,
	SavedClause;

extern Boolean
	**PartialOrder,
	**Compatible;

extern Ordering
	**RecursiveLitOrders;

extern int
	NRecLitClause,
	NRecLitDef;

extern VarInfo
	*Variable;

extern Var
	*DefaultVars;

extern TypeInfo
	*Type;

extern Const
	*Value,
	BoundValue,
	*FnValue,
	FnDefault;

extern Tuple		*Found;
extern int		NFound;

extern Alternative	*ToBeTried;
extern int		NToBeTried;

extern PossibleLiteral	*Possible;
extern int		NPossible,
			NDeterminate;

extern int		*Scheme,
			NScheme;
@//E*O*F Release2/Src/extern.i//
chmod u=rw,g=r,o=r Release2/Src/extern.i
 
echo x - Release2/Src/finddef.c
sed 's/^@//' > "Release2/Src/finddef.c" <<'@//E*O*F Release2/Src/finddef.c//'
/******************************************************************************/
/*									      */
/*	All the stuff for trying possible next literals, growing clauses      */
/*	and assembling definitions					      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"


int	FalsePositive,
	FalseNegative;


void  FindDefinition(Relation R)
/*    --------------  */
{
    int		Size, i, TargetPos, FirstDefR;
    Clause	C;

    Target = R;
    NCl = 0;

    printf("\n----------\n%s:\n", R->Name);

    /*  Reorder the relations so that the target relation comes first  */

    FirstDefR = ( RelnOrder[2] == GTVAR ? 4 : 2 );

    for ( TargetPos = FirstDefR ; RelnOrder[TargetPos] != Target ; TargetPos++ )
	;

    for ( i = TargetPos ; i > FirstDefR ; i-- )
    {
	RelnOrder[i] = RelnOrder[i-1];
    }
    RelnOrder[FirstDefR] = Target;

    /*  Generate initial tuples and make a copy  */

    OriginalState(R);

    StartClause = StartDef;

    Size = StartDef.NTot+1;
    StartClause.Tuples = Alloc(Size, Tuple);
    memcpy(StartClause.Tuples, StartDef.Tuples, Size*sizeof(Tuple));

    NRecLitDef = 0;

    FalsePositive = 0;
    FalseNegative = StartDef.NTot;

    R->Def = Alloc(100, Clause);

    while ( StartClause.NTot )
    { 
	if ( ! (R->Def[NCl] = FindClause()) ) break;

	R->Def[NCl++][NLit] = Nil;

	if ( NCl % 100 == 0 )
	{
	    Realloc(R->Def, NCl+100, Clause);
	}

	NRecLitDef += NRecLitClause;
    }
    R->Def[NCl] = Nil;

    if ( FalsePositive || FalseNegative )
    {
	printf("\n***  Warning: the initial clauses\n");

	if ( FalsePositive )
	{
	    printf("***  give an incorrect value for %d tuple%s\n",
		FalsePositive, Plural(FalsePositive));
	}

	if ( FalseNegative )
	{
	    printf("***  give no value for %d tuple%s\n",
		FalseNegative, Plural(FalseNegative));
	}
    }

    /*SiftClauses();*/

    /*  Add default clause if required  */

    if ( FnDefault )
    {
	C = Target->Def[NCl] = Alloc(2, Literal);

	C[0] = AllocZero(1, struct _lit_rec);
	C[1] = Nil;

	C[0]->Rel  = EQCONST;
	C[0]->Sign = true;
	C[0]->Args = AllocZero(6, Var);

	C[0]->Args[1] = Target->Arity;
	SaveParam(&C[0]->Args[2], &FnDefault);

	printf("\nDefault clause ");
	PrintClause(Target, Target->Def[NCl], false);

	Target->Def[++NCl] = Nil;
    }

    Target->Def[NCl] = Nil;

    if ( SIMPLIFY && ( FalsePositive || FalseNegative ) )
    {
	SimplifyClauses();
    }

    SiftClauses();

    printf("\nFinal theory (after literal and clause pruning):\n");
    PrintDefinition(R);

    pfree(StartClause.Tuples);
    pfree(StartDef.Tuples);

    /*  Restore original relation order  */

    for ( i = FirstDefR ; i < TargetPos ; i++ )
    {
	RelnOrder[i] = RelnOrder[i+1];
    }
    RelnOrder[TargetPos] = Target;
}



Clause  FindClause()
/*      ----------  */
{
    Tuple	Case, *TSP;
    Literal	L;

    InitialiseClauseInfo();

    GrowNewClause();

    if ( NLit > 1 && Current.NPos > 0 ) PruneNewClause();

    /*  Make sure accuracy criterion is satisfied  */

    if ( ! NLit || Current.NOrigPos+1E-3 < MINACCURACY * Current.NOrigTot )
    {
	if ( NLit )
	{
	    Verbose(1)
		printf("\nClause too inaccurate (%d/%d)\n",
		       Current.NOrigPos, Current.NOrigTot);
	}

	pfree(NewClause);
	return Nil;
    }

    /*  Exclude clause that simply binds output variable  */

    if ( NLit == 1 )
    {
	L = NewClause[0];
	if ( L->Rel == EQCONST && L->Args[1] == Target->Arity )
	{
	    if ( FnDefault && Current.NOrigTot <= 2 )
	    {
		printf("\nClause discarded -- would pre-empt default\n");
		return Nil;
	    }
	    else
	    {
		FnDefault = 0;
	    }
	}
    }

    FalsePositive += Current.NOrigTot - Current.NOrigPos;
    FalseNegative -= Current.NOrigTot;

    /*  Set flags for positive covered tuples  */

    ClearFlags;

    for ( TSP = Current.Tuples ; Case = *TSP ; TSP++ )
    {
	SetFlag(Case[0], TrueBit);
    }

    if ( Current.Tuples != StartClause.Tuples )
    {
	FreeTuples(Current.Tuples, true);
    }

    /*  Copy all negative tuples and uncovered positive tuples  */

    StartClause.NTot = StartClause.NPos = 0;

    for ( TSP = StartClause.Tuples ; Case = *TSP ; TSP++ )
    {
	if ( ! TestFlag(Case[0], TrueBit) )
	{
	    StartClause.Tuples[StartClause.NTot++] = Case;
	}
    }
    StartClause.Tuples[StartClause.NTot] = Nil;

    StartClause.NOrigPos = StartClause.NPos;
    StartClause.NOrigTot = StartClause.NTot;

    StartClause.BaseInfo = Info(StartClause.NPos, StartClause.NTot);
    Current = StartClause;

    Verbose(1)
    {
	printf("\nClause %d: ", NCl);
	PrintClause(Target, NewClause, true);
    }

    return NewClause;
}



void  ExamineLiterals()
/*    ---------------  */
{
    Relation	R;
    int		i, Relns=0;

    NPossible = NDeterminate = 0;

    /*  If this is not the first literal, review coverage and check
	variable orderings, identical variables etc.  */

    if ( NLit != 0 )
    {
	/*CheckOriginalCaseCover();*/
	ExamineVariableRelationships();
    }

    if ( Current.NPos )
    {
	/*  Function value already bound  */

	MaxPossibleGain = Current.NPos * Current.BaseInfo;
	AvailableBits = Encode(Current.NOrigPos) - ClauseBits;
    }
    else
    {
	MaxPossibleGain = Current.NTot * Current.BaseInfo;
	AvailableBits = Except(FnRange * Current.NOrigTot, Current.NOrigTot)
			- ClauseBits;
    }

    Verbose(1)
    {
	printf("\nState (%d/%d", Current.NPos, Current.NTot);
	if ( Current.NTot != Current.NOrigTot )
	{
	    printf(" [%d/%d]", Current.NOrigPos, Current.NOrigTot);
	}
	printf(", %.1f bits available", AvailableBits);

	if ( NWeakLits )
	{
	    Verbose(2)
		printf(", %d weak literal%s", NWeakLits, Plural(NWeakLits));
	}
	printf(")\n");

	Verbose(4)
	    PrintTuples(Current.Tuples, Current.MaxVar);
    }

    /*  Find possible literals for each relation  */

    ForEach(i, 0, MaxRel)
    {
	R = RelnOrder[i];

	ExploreArgs(R, true);

	if ( R->NTrialArgs ) Relns++;
    }

    /*  Evaluate them  */

    AlterSavedClause = Nil;
    Verbose(2) putchar('\n');

    for ( i = 0 ; i <= MaxRel && BestLitGain < MaxPossibleGain ; i++ )
    {
	R = RelnOrder[i];
	if ( ! R->NTrialArgs ) continue;

	R->Bits = Log2(Relns) + Log2(R->NTrialArgs+1E-3);
	if ( NEGLITERALS || Predefined(R) ) R->Bits += 1.0;

	if ( R->Bits - Log2(NLit+1.001-NDetLits) > AvailableBits )
	{
	    Verbose(2)
	    {
		printf("\t\t\t\t[%s requires %.1f bits]\n", R->Name, R->Bits);
	    }
	}
	else
	{
	    ExploreArgs(R, false);

	    Verbose(2)
		printf("\t\t\t\t[%s tried %d/%d] %.1f secs\n",
		       R->Name, R->NTried, R->NTrialArgs, CPUTime());
	}
    }
}



void  GrowNewClause()
/*    -------------                */
{
    Literal	L;
    int		i, OldNLit;
    Boolean	Progress=true;
    float	Accuracy, ExtraBits;

    while ( Progress && Current.NPos < Current.NTot )
    {
	ExamineLiterals();

	/*  If have noted better saveable clause, record it  */

	if ( AlterSavedClause )
	{
	    Realloc(SavedClause, NLit+2, Literal);
	    ForEach(i, 0, NLit-1)
	    {
		SavedClause[i] = NewClause[i];
	    }
	    SavedClause[NLit]   = AllocZero(1, struct _lit_rec);
	    SavedClause[NLit+1] = Nil;

	    SavedClause[NLit]->Rel      = AlterSavedClause->Rel;
	    SavedClause[NLit]->Sign     = AlterSavedClause->Sign;
	    SavedClause[NLit]->Args     = AlterSavedClause->Args;
	    SavedClause[NLit]->Bits     = AlterSavedClause->Bits;
	    SavedClause[NLit]->WeakLits = 0;

	    SavedClauseCover    = AlterSavedClause->PosCov;
	    SavedClauseAccuracy = AlterSavedClause->PosCov /
					  (float) AlterSavedClause->TotCov;

	    Verbose(1)
	    {
		printf("\n\tSave clause ending with ");
		PrintLiteral(SavedClause[NLit]);
		printf(" (cover %d, accuracy %d%%)\n",
		       SavedClauseCover, (int) (100*SavedClauseAccuracy));
	    }

	    pfree(AlterSavedClause);
	}

	if ( NDeterminate && BestLitGain < DETERMINATE * MaxPossibleGain )
	{
	    ProcessDeterminateLiterals();
	}
	else
	if ( NPossible )
	{
	    /*  At least one gainful literal  */

	    NewClause[NLit] = L = SelectLiteral();
	    if ( ++NLit % 100 == 0 ) Realloc(NewClause, NLit+100, Literal);

	    ExtraBits = L->Bits - Log2(NLit-NDetLits+1E-3);
	    ClauseBits += Max(ExtraBits, 0);

	    Verbose(1)
	    {
		printf("\nBest literal ");
		PrintLiteral(L);
		printf(" (%.1f bits)\n", L->Bits);
	    }

	    /*  Check whether should regrow clause  */

	    if ( L->Rel != Target && AllLHSVars(L) &&
		 Current.MaxVar > Target->Arity && ! AllDeterminate() )
	    {
		OldNLit = NLit;
		NLit = 0;
		ForEach(i, 0, OldNLit-1)
		{
		    if ( AllLHSVars(NewClause[i]) )
		    {
			NewClause[NLit++] = NewClause[i];
		    }
		}
		NewClause[NLit] = Nil;

		RecoverState(NewClause, false);

		Verbose(1)
		{
		    printf("\n[Regrow clause] ");
		    PrintClause(Target,NewClause, false);
		}
		GrowNewClause();
		return;
	    }

	    NWeakLits = L->WeakLits;

	    if ( L->Rel == Target ) AddOrders(L);

	    NewState(L, Current.NTot);

	    if ( L->Rel == Target ) NoteRecursiveLit(L);
	}
	else
	{
	    Verbose(1) printf("\nNo literals\n");

	    Progress = Recover();
	}
    }
    NewClause[NLit] = Nil;

    /*  Finally, see whether saved clause is better  */

    /*CheckOriginalCaseCover();*/
    Accuracy = Current.NOrigPos / (float) Current.NOrigTot;
    if ( SavedClause &&
	 ( SavedClauseAccuracy > Accuracy ||
	   SavedClauseAccuracy == Accuracy &&
	   SavedClauseCover > Current.NOrigPos ||
	   SavedClauseAccuracy == Accuracy &&
	   SavedClauseCover == Current.NOrigPos &&
	   CodingCost(SavedClause) < CodingCost(NewClause) ) )
    {
	Verbose(1) printf("\n[Replace by saved clause]\n");
printf("coding costs %g %g\n", CodingCost(SavedClause), CodingCost(NewClause));
	RecoverState(SavedClause, true);
	/*CheckOriginalCaseCover();*/
    }
}


    InitialiseClauseInfo()
/*  --------------------  */
{
    Var V;

    /*  Initialise everything for start of new clause  */

    NewClause = Alloc(100, Literal);

    Current = StartClause;

    NLit = NDetLits = NWeakLits = NRecLitClause = 0;

    NToBeTried = 0;
    AnyPartialOrder = false;

    AllTuples  = Current.NTot * FnRange;
    ClauseBits = 0;

    ForEach(V, 1, Target->Arity)
    {
	Variable[V]->Depth   = 0;
	Variable[V]->Type    = Target->Type[V];
	Variable[V]->TypeRef = Target->TypeRef[V];
    }

    memset(Barred, false, MAXVARS+1);

    SavedClause = Nil;
    SavedClauseAccuracy = SavedClauseCover = 0;

    MAXRECOVERS = MAXALTS;
}



Boolean  AllLHSVars(Literal L)
/*       ----------  */
{
    Var V;

    ForEach(V, 1, L->Rel->Arity)
    {
	if ( L->Args[V] > Target->Arity ||
	     L->Args[V] == Target->Arity && ! L->Sign ) return false;
    }

    return true;
}



	/*  See whether all literals in clause are determinate or involve
	    only LHS variables  */

Boolean  AllDeterminate()
/*       --------------  */
{
    int i;

    ForEach(i, 0, NLit-2)
    {
	if ( NewClause[i]->Sign != 2 && ! AllLHSVars(NewClause[i]) ) return false;
    }

    return true;
}


	/*  Find the coding cost for a clause  */

float	CodingCost(Clause C)
/*      ----------  */
{
    float	SumBits=0, Contrib;
    int		Lits=0;

    while ( *C )
    {
	Lits++;
	if ( (*C)->Rel != EQVAR &&
	     (Contrib = (*C)->Bits - Log2(Lits)) > 0 )
	{
	    SumBits += Contrib;
	}
	C++;
    }

    return SumBits;
}
@//E*O*F Release2/Src/finddef.c//
chmod u=rw,g=r,o=r Release2/Src/finddef.c
 
echo x - Release2/Src/global.c
sed 's/^@//' > "Release2/Src/global.c" <<'@//E*O*F Release2/Src/global.c//'
#include "defns.i"

/******************************************************************************/
/*									      */
/*	Parameters set by options and variables accessible to many routines   */
/*									      */
/******************************************************************************/


Boolean
	NEGLITERALS   = true,	/* negated literals ok */
	NEGEQUALS     = true,	/* negated equality literals ok */
/*	UNIFORMCODING = false,	/* uniform coding of literals */
	SIMPLIFY      = true,	/* post-prune clauses */

	MissingVals   = false,	/* missing values in input? */
	AnyPartialOrder,	/* quick check to rule out recursive lits */
	*Barred,		/* duplicate variables */
	*VarUsed,		/* variable used (for QuickPrune) */
	OutOfWorld;		/* flag ^ constant */


float
	MINACCURACY = 0.8,	/* minimum acceptable clause accuracy */
	MINALTFRAC  = 0.8,	/* fraction of best gain required for backup */
	DETERMINATE = 0.8;  	/* use determinate literals unless a literal
				   with this fraction of max possible gain */

int
	MAXVARS	    = 52,	/* max number of variables */
	MAXARGS     = 5,	/* max arity of any relation */
	MAXWEAKLITS = 3,	/* max weak literals in sequence */

	MAXPOSSLIT  = 5,	/* 1 + max backups from single state */
	MAXALTS     = 20,	/* max simultaneous backups */
	MAXRECOVERS,		/* max total backups */
	MAXVARDEPTH = 4,	/* max depth of var in literal */
	MAXTUPLES   = 100000,	/* max number of tuples */
	VERBOSITY   = 1,	/* level of output */

	MaxConst    = 0,	/* no. constants */
	MaxType	    = 0,	/* no. types */
	MaxRel	    = 0,	/* highest relation no */

	FnRange,		/* number of possible values of function */
	AllTuples,		/* effective size of universe */
	NCl,			/* current clause number */
	NLit,			/* current literal number */
	NDetLits,		/* number of determinate lits in clause */
	NWeakLits,		/* current weak lits in sequence */
	SavedClauseCover;	/* coverage of saved clause  */

char
	**ConstName,		/* names of all discrete constants */
	*Flags = Nil;		/* flag bits for original tuples */


Relation
	*Reln,			/* relations */
	*RelnOrder,      	/* order to try relations */
	Target;			/* relation being induced */


State
	StartDef,		/* at start of definition */
	StartClause,		/* at start of clause */
	Current,		/* current state */
	New;			/* possible next state */


float
	*LogFact = Nil,		/* LogFact[i] = log2(i!) */
	MaxPossibleGain,
	ClauseBits,		/* bits used so far in this clause */
	AvailableBits,		/* bits available for this clause */
	SavedClauseAccuracy;	/* accuracy of saved clause */


Clause
	NewClause,		/* clause being constructed */
	SavedClause;		/* best shorter clause discovered
				   while developing current clause */

PossibleLiteral
	AlterSavedClause;	/* last literal of saved clause */


Boolean
	**PartialOrder,		/* partial orders on variables*/
	**Compatible;		/* Compatible[i][j] = true if types i, j have
				   at least one common value  */

Ordering
	**RecursiveLitOrders;	/* pointers to orders in recursive lits  */

int
	NRecLitClause,		/* number of recursive lits in the new clause */
	NRecLitDef;		/* ditto in definition so far */

VarInfo
	*Variable;		/* variables */

Var
	*DefaultVars;		/* default variable list */

TypeInfo
	*Type;			/* types */

Const
	*Value = Nil,		/* current variable bindings */
	BoundValue,
	*FnValue = Nil,		/* target value for functions */
	FnDefault;		/* default value or 0 */

Tuple		*Found;		/* join */
int		NFound;		/* number of tuples in join */

Alternative	*ToBeTried;	/* backup points */
int		NToBeTried;

PossibleLiteral	*Possible;	/* possible literals */
int		NPossible,
		NDeterminate;

int	*Scheme,		/* tests for recursive soundness */
	NScheme;
@//E*O*F Release2/Src/global.c//
chmod u=rw,g=r,o=r Release2/Src/global.c
 
echo x - Release2/Src/input.c
sed 's/^@//' > "Release2/Src/input.c" <<'@//E*O*F Release2/Src/input.c//'
/******************************************************************************/
/*									      */
/*	All input routines						      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"

#define  Space(s)	(s == ' ' || s == '\t')
#define  SkipComment	while ( ( c = getchar() ) != '\n' )


char    Name[200],
	SectionDelim;
Const   *SortedConst = Nil;
Boolean	ContinuousVars = false;


/******************************************************************************/
/*									      */
/*	Routines for reading types					      */
/*									      */
/******************************************************************************/



Boolean  ReadType()
/*       --------  */
{
    int		i;
    char	Delim;
    TypeInfo	T;
    Boolean	FirstTime=true, RecordTheoryConst;
    Const	C;

    Delim = ReadName(Name);

    if ( ! *Name ) return false;
    else
    if ( Delim != ':' ) Error(1, Name);

    T = AllocZero(1, struct _type_rec);

    T->NValues = T->NTheoryConsts = 0;
    T->Value   = T->TheoryConst   = Nil;

    if ( Name[0] == '*' )	/* order specified by user */
    {
        T->FixedPolarity = true;
	T->Ordered = true;
        T->Name = CopyString(Name+1);
    }
    else
    if ( Name[0] == '#' )	/* specified to be unordered */
    {
        T->FixedPolarity = true;
        T->Ordered = false;
        T->Name = CopyString(Name+1);
    }
    else
    {
        T->FixedPolarity = false;
	T->Name = CopyString(Name);
    }

    /*  Read first name  */

    while ( (Delim = ReadName(Name)) == '\n' )
	;

    if ( ! Name ) Error(1, Name);

    /*  Check for continuous type  */

    if ( ! strcmp(Name, "continuous") )
    {
        T->Continuous = true;
	T->FixedPolarity = true;
	T->Ordered = false; /* Never match on continuous value - hence
			       no point in checking for orders on it */
	ContinuousVars = true;
    }
    else
    {
	/*  Discrete type, read the values  */

        T->Continuous = false;

	do
	{
	    while ( ( FirstTime ? ! (FirstTime = false) :
				    (Delim = ReadName(Name)) ) &&
		    Delim == '\n' )
			;

	    if ( Name[0] == '*' )
	    {
		/*  Theory constant  */

		RecordTheoryConst = true;

		for ( i = 0 ; Name[i++] ; )
		{
		    Name[i-1] = Name[i];
		}
	    }
	    else
	    {
		RecordTheoryConst = false;
	    }
		
	    if ( T->NValues % 100 == 0 )
	    {
		Realloc(T->Value, T->NValues+100, Const);
	    }

	    C = T->Value[ T->NValues++ ] = FindConstant(Name, false);

	    /*  Check for duplicate constants  */

	    ForEach(i, 0, T->NValues-2)
	    {
		if ( T->Value[i] == C ) Error(7, Name, T->Name);
	    }

	    if ( RecordTheoryConst )
	    {
		if ( T->NTheoryConsts % 10 == 0 )
		{
		    Realloc(T->TheoryConst, T->NTheoryConsts+10, Const);
		}

		T->TheoryConst[ T->NTheoryConsts++ ] = C;
	    }
	}
	while ( Delim == ',' );
    }

    if ( Delim != '.' ) Error(2, Name, T->Name);
    ReadToEOLN;

    /* Enter Type */

    MaxType++;

    if ( MaxType % 100 == 1 )
    {
        Realloc(Type, MaxType + 100, TypeInfo);
    }

    Type[MaxType] = T;

    return true;
}



void  ReadTypes()
/*    ---------  */
{
    int		i, j;
    TypeInfo	T;

    /*  Record names for missing value and out-of-range  */

    FindConstant("?", false);
    FindConstant("^", false);

    /*  Read all type definitions  */

    while ( ReadType() )
	;

    /*  Generate collating sequences  */

    ForEach(i, 1, MaxType)
    {
	T = Type[i];

	if ( T->Continuous ) continue; /* Skip continuous type */

	T->CollSeq = Alloc(MaxConst+1, int);

	ForEach(j, 0, MaxConst)
	{
	    T->CollSeq[j] = 0;
	}

	ForEach(j, 0, T->NValues-1)
	{
	    T->CollSeq[T->Value[j]] = j+1;
	}

	T->CollSeq[MISSING_DISC] = T->NValues+1;
    }
}



/******************************************************************************/
/*									      */
/*	Routines for reading tuples and relations			      */
/*									      */
/******************************************************************************/



Tuple  ReadTuple(Relation R)
/*     ---------  */
{
    char	Delim;
    int		N, i;
    Tuple	T;

    N = R->Arity;

    if ( (Delim = ReadName(Name)) == '.' || Delim == ';' )
    {
	return Nil;
    }

    T = Alloc(N+1, Const);

    ForEach(i, 1, N)
    {
	if ( i > 1 )
	{
	    Delim = ReadName(Name);
	}

	if ( R->TypeRef[i]->Continuous )
	{
	    if ( ! strcmp(Name,"?") )
	    {
	        FP(T[i]) = MISSING_FP;
		MissingVals = true;
	    }
	    else
	    {
	        FP(T[i]) = atof(Name);

		if ( FP(T[i]) == MISSING_FP )
		{
		    printf("An input continuous values is equal to the\n");
		    printf("magic number used to designate missing values.\n");
		    printf("Change the definition of MISSING_FP in defns.i\n");
		    printf("and recompile.\n");
		    exit(1);
	        }
	    }
	}
	else
	{
	    if ( ! strcmp(Name,"?") )
	    {
	        T[i] = MISSING_DISC;
		MissingVals = true;
	    }
	    else
	    {
		T[i] = FindConstant(Name, true);
	    }
	}
    }

    if ( Delim != ':' && Delim != '\n' ) ReadToEOLN;

    ForEach(i, 1, N)
    {
	if ( ! R->TypeRef[i]->Continuous && T[i] != OUT_OF_RANGE &&
	     ! R->TypeRef[i]->CollSeq[T[i]] ) 
	{
            Error(4, ConstName[T[i]], Type[R->Type[i]]->Name);
	}
    }

    return T;
}



Tuple  *ReadTuples(Relation R)
/*      ----------  */
{
    Tuple	T, *TheTuples=Nil;
    int		ND=0;

    while ( T = ReadTuple(R) )
    {
	T[0] = 0;

        if ( ND % 100 == 0 ) 
	{
	    Realloc(TheTuples, ND+101, Tuple);
	}
	TheTuples[ND++] = T;
    }

    TheTuples[ND] = Nil;

    return TheTuples;
}



Relation  ReadRelation()
/*        ------------  */
{
    Relation	R;
    char	Delim, c;
    int		NArgs=0, Key[100], NKeys=0, i, t;

    if ( ReadName(Name) != '(' ) return Nil;

    printf("\nRelation %s\n", Name);

    /*  Create a new record with all zero counts  */

    R = AllocZero(1, struct _rel_rec);

    if ( Name[0] == '*' )
    {
	/*  Background relation  */

        R->PossibleTarget = false;
        R->Name = CopyString(Name+1);
    }
    else
    {
        R->PossibleTarget = true;
	R->Name = CopyString(Name);
    }

    do
    {
	NArgs++;

        Realloc(R->Type, NArgs+1, int);
        Realloc(R->TypeRef, NArgs+1, TypeInfo); 
	Delim = ReadName(Name);
        t = FindType(Name);
	R->Type[NArgs] = t; 
        R->TypeRef[NArgs] = Type[t]; 
    }
    while ( Delim != ')' );

    R->Arity = NArgs;
    if ( NArgs > MAXARGS ) MAXARGS = NArgs;
    if ( R->PossibleTarget && NArgs > MAXVARS ) MAXVARS = NArgs;

    /*  Read and store access keys  */
    do
    {
	do
	{
	    c = getchar();
	}
	while ( Space(c) ) ;

	if ( c != '\n' )
	{
	    Key[NKeys] = 0;

	    ForEach(i, 1, NArgs)
	    {
		if ( c == '-' ) Key[NKeys] |= (1 << i);
		c = getchar();
	    }
	    NKeys++;
	}
    }
    while ( c == '/');

    R->NKeys = NKeys;
    if ( NKeys )
    {
	R->Key   = Alloc(NKeys, int);
	memcpy(R->Key, Key, NKeys*sizeof(int));
    }

    R->Pos      = ReadTuples(R);
    R->PosIndex = MakeIndex(R->Pos, NArgs, R);

    if ( SectionDelim == '.' )
    {
	R->Neg = Nil;
    }
    else
    {
	printf("\n*** Negative tuples for %s ignored by fFOIL\n", R->Name);
	R->Neg = ReadTuples(R);
    }

    R->BinSym = SymmetryCheck(R);

    DuplicateTuplesCheck(R);

    UnequalArgsCheck(R);
    return R;
}



void  ReadRelations()
/*    -------------   */
{
    int		i, j, Next, Best, PosSize, WorldSize;
    Relation	R;
    Tuple	*T;
    Boolean	*Waiting;
    float	*Imbalance;

    while ( R = ReadRelation() )
    {
	/*  Make sure room for one more  */

        if ( ++MaxRel % 10 == 0 ) 
        {
            Realloc(Reln, MaxRel+10, Relation);
        }

	Reln[MaxRel] = R;

	Verbose(4)
	{
	    if ( Reln[MaxRel]->BinSym )
	    {
		printf("    is binary symmetric\n");
	    }

	    for ( T = Reln[MaxRel]->Pos ; *T ; T++ )
	    {
		PrintTuple(*T, Reln[MaxRel]->Arity,
			   Reln[MaxRel]->TypeRef, Reln[MaxRel]->Neg != Nil);
	    }
		
	    if ( Reln[MaxRel]->Neg )
	    {
		for ( T = Reln[MaxRel]->Neg ; *T ; T++ )
		{
		    PrintTuple(*T, Reln[MaxRel]->Arity,
			       Reln[MaxRel]->TypeRef, true);
		}
	    }
	}
    }

    /*  Now put the relations into the order in which they should be tried.
	The idea is to put lower arity relations earlier to maximise the
	effect of pruning.  Relations of the same arity are resolved by
	preferring relations with higher information  */

    RelnOrder = Alloc(MaxRel+1, Relation);
    Waiting = Alloc(MaxRel+1, Boolean);
    Imbalance = Alloc(MaxRel+1, float);

    memset(Waiting, true, MaxRel+1);
    ForEach(i, 4, MaxRel)
    {
	R = Reln[i];
	PosSize = Number(R->Pos);

	if ( R->Neg )
	{
	    WorldSize = PosSize + Number(R->Neg);
	}
	else
	{
	    WorldSize = 1;
	    ForEach(j, 1, Reln[i]->Arity)
	    {
		if ( ! R->TypeRef[j]->Continuous )
		{
		    WorldSize *= R->TypeRef[j]->NValues;
		}
	    }
	}

	Imbalance[i] = fabs(0.5 - PosSize / (float) WorldSize);
    }

    RelnOrder[0] = Reln[1];
    RelnOrder[1] = Reln[0];
    RelnOrder[2] = Reln[3];
    RelnOrder[3] = Reln[2];
    Next = ( ContinuousVars ? 4 : 2 );
    
    while ( true )
    {
	Best = -1;

	ForEach(i, 4, MaxRel)
	{
	    if ( Waiting[i] &&
		 ( Best < 0 ||
		   Reln[i]->Arity < Reln[Best]->Arity ||
		   Reln[i]->Arity == Reln[Best]->Arity
		   && Imbalance[i] < Imbalance[Best] ) )
	    {
		Best = i;
	    }
	}

	if ( Best < 0 ) break;
	RelnOrder[Next++] = Reln[Best];
	Waiting[Best] = false;
    }
    MaxRel = Next-1;

    pfree(Waiting);
    pfree(Imbalance);
}


	/*  Find a type by name  */

int  FindType(char *N)
/*   --------  */
{
    int i;

    ForEach(i, 1, MaxType)
    {
	if ( ! strcmp(N, Type[i]->Name) ) return i;
    }

    Error(5, N);
    return 0; /* keep lint happy */
}



/******************************************************************************/
/*                                                                            */
/*	DuplicateTuplesCheck(R) - check for duplicate tuples in R             */
/*                                                                            */
/******************************************************************************/


void  DuplicateTuplesCheck(Relation R)
/*    --------------------  */
{
    int		i, j, k, N, NPos, NNeg;
    Tuple	*PosCopy, *NegCopy, PosTuple, NegTuple;
    Boolean	MutualDuplicate;

    /* First copy the positive tuples and check number of duplicates */

    NPos = Number(R->Pos);

    PosCopy = Alloc(NPos+1, Tuple);
    ForEach(i, 0, NPos)
    {
	PosCopy[i] = (R->Pos)[i];
    }

    N = R->Arity;

    if ( R->PosDuplicates = CountDuplicates(PosCopy,N,0,NPos-1) )
    {
	printf("    (warning: contains duplicate positive tuples)\n");
    }

    /* If there are neg tuples, check for duplicates and mutual duplicates */

    if ( R->Neg )
    {
        NNeg = Number(R->Neg);
        NegCopy = Alloc(NNeg+1, Tuple);
	ForEach(i, 0, NNeg)
	{
	    NegCopy[i] = (R->Neg)[i];
	}

	if ( CountDuplicates(NegCopy,N,0,NNeg-1) )
        {
	    printf("    (warning: contains duplicate negative tuples)\n");
	}


	/* Existence check for mutual duplicates */

	MutualDuplicate = false;
	i = j = 0;

	while( i < NPos && j < NNeg )
	{
	    PosTuple = PosCopy[i];
	    NegTuple = NegCopy[j];

	    for ( k = 1 ; k <= N && PosTuple[k] == NegTuple[k] ; k++ )
		;

	    if ( k > N ) /* tuples are duplicates */
	    {
	        MutualDuplicate = true;
		break;
	    }
	    else
	    if ( PosTuple[k] < NegTuple[k] )
	    {
		i++;
	    }
	    else
	    {
		j++;
	    }
	}

	if ( MutualDuplicate ) 
	{
	    printf("    (warning: contains tuples that are both ");
	    printf("positive and negative)\n");
	}

	pfree(NegCopy);
    }

    pfree(PosCopy);
}



/******************************************************************************/
/*                                                                            */
/*	CountDuplicates(T,N,left,right) - count the number of duplicate	      */
/*	    tuples in T between left and right.       			      */
/*		Sorts tuples on order given by comparison of Const type.      */
/*		N.B. This comparison is used even for continuous values as    */
/*		only checking for duplicates. 				      */
/*                                                                            */
/******************************************************************************/


int  CountDuplicates(Tuple *T, int N, int Left, int Right)
/*   ---------------  */
{
    register int	i, j, last, first, swap, count=0;
    register Tuple	temp, comp, other;

    if ( Left >= Right ) return 0;

    temp = T[Left];
    T[Left] = T[swap=(Left+Right)/2];
    T[swap] = temp;

    last = Left;

    comp = T[Left];

    for ( i = Left + 1; i <= Right; i++ )
    {
        other = T[i];
	for( j = 1 ; j <= N && other[j] == comp[j] ; j++ )
	    ;

        if ( j > N || other[j] < comp[j] ) /* other <= comp */
	{
	    temp = T[++last];
	    T[last] = T[i];
	    T[i] = temp;
	}
    }

    temp = T[Left];
    T[Left] = T[last];
    T[last] = temp;

    first = last;

    for ( i = last - 1; i >= Left ; i-- )
    {
        other = T[i];
	for ( j = 1 ; j <= N && other[j] == comp[j] ; j++ )
	    ;

	if ( j > N ) /* other == comp */
	{
	    temp = T[--first];
	    T[first] = T[i];
	    T[i] = temp;
	    count++;
	}
    }

    count += CountDuplicates(T,N,Left,first-1);
    count += CountDuplicates(T,N,last+1,Right);

    return count;
}



Boolean  SymmetryCheck(Relation R)
/*       -------------    */
{
    Tuple	*TheTuples;
    Boolean	*SymCheck;
    int		i, j, NPos;
    Const	T1, T2;

    if ( R->Arity != 2 ||
	 R->TypeRef[1]->Continuous || R->TypeRef[2]->Continuous )
    {
        return false;
    }

    TheTuples = R->Pos;
    NPos = Number(TheTuples);

    SymCheck = Alloc(NPos, Boolean);
    memset(SymCheck, false, NPos*sizeof(Boolean));

    ForEach(i, 0, NPos-1)
    {
        if ( SymCheck[i] ) continue; 

        T1 = TheTuples[i][1];
        T2 = TheTuples[i][2];
        for ( j = i ;
	      j < NPos && ( T1 != TheTuples[j][2] || T2 != TheTuples[j][1] ) ;
	      j++ )
	    ;

        if ( j == NPos )
	{
	    pfree(SymCheck);
	    return false;
        }
        SymCheck[j] = true;
    }

    pfree(SymCheck);
    return true;
}


	/*  Construct the index for a set of tuples for relation R  */

Index  MakeIndex(Tuple *T, int N, Relation R)
/*     ---------  */
{
    Index	IX;
    Tuple	Case, *Scan;
    int		**Next, Arg, Val, No = 0;

    /*  Allocate storage  */

    IX = Alloc(N+1, int **);
    Next = Alloc(N+1, int *);

    ForEach(Arg, 1, N)
    {
	IX[Arg] = Alloc(MaxConst+1, int *);
	Next[Arg] = AllocZero(MaxConst+1, int);
    }

    for ( Scan = T ; Case = *Scan++ ; )
    {
	ForEach(Arg, 1, N)
	{
	    if ( ! R->TypeRef[Arg]->Continuous )
	        Next[Arg][Case[Arg]]++;
	}
    }

    ForEach(Arg, 1, N)
    {
	ForEach(Val, 1, MaxConst)
	{
	    IX[Arg][Val] = Next[Arg][Val] ? Alloc(Next[Arg][Val]+1, int) : Nil;
	    Next[Arg][Val] = 0;
	}
    }

    /*  Construct the index  */

    for ( Scan = T ; *Scan ; Scan++ )
    {
	ForEach(Arg, 1, N)
	{
	    if ( ! R->TypeRef[Arg]->Continuous )
	    {
	        Val = (*Scan)[Arg];
		IX[Arg][Val][Next[Arg][Val]++] = No;
	    }
	}

	No++;
    }

    /*  Terminate index and free Next  */

    ForEach(Arg, 1, N)
    {
	ForEach(Val, 1, MaxConst)
	{
	    if ( IX[Arg][Val] )
	    {
		IX[Arg][Val][Next[Arg][Val]] = FINISH;
	    }
	}
	pfree(Next[Arg]);
    }
    pfree(Next);

    return IX;
}



/******************************************************************************/
/*									      */
/*	Basic routine -- read a delimited name into string s		      */
/*									      */
/*	  - Embedded spaces are permitted, but multiple spaces are replaced   */
/*	    by a single space						      */
/*	  - Any character escaped by \ is ok				      */
/*	  - Characters after | are ignored				      */
/*									      */
/******************************************************************************/



char  ReadName(char *s)
/*    ---------  */
{
    register char	*Sp = s;
    register int	c;

    /*  Skip to first non-space character  */

    while ( (c = getchar()) != EOF && ( c == '|' || Space(c) ) )
    {
	if ( c == '|' ) SkipComment;
    }

    /*  Return period if no names to read  */

    if ( c == EOF )
    {
	return (SectionDelim = '.');
    }
    else
    if ( c == ';' || c == '.' )
    {
	ReadToEOLN;
	return (SectionDelim = c);
    }

    /*  Read in characters up to the next delimiter  */

    while ( c != ',' && c != '\n' && c != '|' && c != EOF &&
	    c != '(' && c != ')'  && c != ':' && c != '.' )
    {
	if ( c == '\\' ) c = getchar();

	*Sp++ = c;
	if ( c == ' ' )
	    while ( ( c = getchar() ) == ' ' );
	else
	    c = getchar();

	if ( c == '.' ) /* Check for embedded period in number */
	{
	    c = getchar();
	    if (isdigit(c))
	    {
	        *Sp++ = '.';
	    }
	    else
	    {
	        ungetchar(c);
		c = '.';
	    }
	}
    }

    if ( c == '|' )
    {
	SkipComment;
	c = '\n';
    }

    /*  Strip trailing spaces  */

    while ( Sp > s && Space(*(Sp-1)) ) Sp--;
    *Sp++ = '\0';

    return c;
}



	/*  Find a constant using binary chop search  */


Const  FindConstant(char *N, Boolean MustBeThere)
/*     ------------  */
{
    int	i, Hi=MaxConst+1, Lo=1, Differ=1;

    while ( Lo < Hi-1 )
    {
	Differ = strcmp(N, ConstName[SortedConst[i = (Hi + Lo)/2]]);

	if ( ! Differ )		return SortedConst[i];
	else
	if ( Differ > 0 )	Lo = i;
	else			Hi = i;
    }

    if ( MustBeThere ) Error(3, N);

    /*  This is a new constant -- record it  */

    MaxConst++;
    if ( MaxConst % 1000 == 1 )
    {
	Realloc(ConstName, MaxConst+1000, char *);
	Realloc(SortedConst, MaxConst+1000, int);
    }

    Lo++;
    for ( i = MaxConst ; i > Lo ; i-- )
    {
	SortedConst[i] = SortedConst[i-1];
    }
    SortedConst[Lo] = MaxConst;
    ConstName[MaxConst] = CopyString(N);

    return MaxConst;
}



	/*  Check whether different types are compatible, i.e.
	    share at least one common value  */

void  CheckTypeCompatibility()
/*    ----------------------  */
{
    int T1, T2;

    Compatible = Alloc(MaxType+1, Boolean *);
    ForEach(T1, 1, MaxType)
    {
	Compatible[T1] = Alloc(MaxType+1, Boolean);
    }

    Verbose(2) putchar('\n');

    ForEach(T1, 1, MaxType)
    {
	Compatible[T1][T1] = true;

	ForEach(T2, T1+1, MaxType)
	{
	    Compatible[T1][T2] = Compatible[T2][T1] =
	        ( Type[T1]->Continuous || Type[T2]->Continuous ) ?
		false:
		CommonValue(Type[T1]->NValues, Type[T1]->Value,
		            Type[T2]->NValues, Type[T2]->Value);

	    Verbose(2)
	    {
		printf("Types %s and %s %s compatible\n",
			Type[T1]->Name, Type[T2]->Name,
			Compatible[T1][T2] ? "are" : "are not");
	    }
	}
    }
}


Boolean  CommonValue(int N1, Const *V1, int N2, Const *V2)
/*       -----------  */
{
    int i, j;

    ForEach(i, 0, N1-1)
    {
        ForEach(j, 0, N2-1)
	{
	    if ( V1[i] == V2[j] ) return true;
	}
    }

    return false;
}



int  Number(Tuple *T)
/*   ------  */
{
    int Count=0;

    if ( ! T ) return 0;

    while ( *T++ )
    {
	Count++;
    }

    return Count;
}



char  *CopyString(char *s)
/*     ----------  */
{
    char *new;
    int l;

    l = strlen(s) + 1;
    new = Alloc(l, char);
    memcpy(new, s, l);

    return new;
}



void  Error(int n, char *s1, char *s2)
/*    -----  */
{
    switch ( n )
    {
        case 1:	printf("Illegal delimiter after %s\n", s1);
		exit(1);

	case 2:	printf("Something wrong after %s in type %s\n", s1, s2);
		exit(1);

	case 3: printf("Undeclared constant %s\n", s1);
		exit(1);

	case 4: printf("Constant %s is not of type %s\n", s1, s2);
		exit(1);

	case 5: printf("Undeclared type %s\n", s1);
		exit(1);

	case 6: printf("Cannot use CWA for %s (continuous types)\n", s1);
		exit(1);

	case 7: printf("Type %s contains duplicate constant %s\n", s2, s1);
		exit(1);
    }
}



    /*  Check for arguments that cannot be equal  */

void  UnequalArgsCheck(Relation R)
/*    ----------------  */
{
    Var	S, F;

    R->ArgNotEq = AllocZero(ArgPair(R->Arity,R->Arity), Boolean);
    ForEach(S, 2, R->Arity)
    {
	ForEach(F, 1, S-1)
	{
	    R->ArgNotEq[ ArgPair(S,F) ] = NeverEqual(R->Pos, F, S);
	}
    }
}



Boolean  NeverEqual(Tuple *T, Var F, Var S)
/*       ----------  */
{
    Tuple	Case;

    while ( Case = *T++ )
    {
	if ( Case[F] == Case[S] ) return false;
    }

    return true;
}
@//E*O*F Release2/Src/input.c//
chmod u=rw,g=r,o=r Release2/Src/input.c
 
echo x - Release2/Src/interpret.c
sed 's/^@//' > "Release2/Src/interpret.c" <<'@//E*O*F Release2/Src/interpret.c//'
/******************************************************************************/
/*									      */
/*	Routines for evaluating a definition on a case.  Used both during     */
/*	pruning and when testing definitions found			      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"

#define  HighestVar	Current.MaxVar	/* must be set externally! */

Boolean	RecordArgOrders = false,	/* flag set by prune */
	MultipleValues;			/* used in SimplifyClause() */


Boolean  CheckRHS(Clause C)
/*       --------  */
{
    Relation	R;
    Tuple	Case, *Bindings, *Scan, LastSuccess=0;
    Literal	L;
    int		i, j, N;
    Var		*A, V, W, FnVar=0;
    Const	KVal, *CopyValue;
    float	VVal, WVal;
    Ordering	ThisOrder, PrevOrder;
    Boolean	SomeOrder=false, NowBound=false, Stop=false;

    if ( ! (L = C[0]) ) return true;

    R = L->Rel;
    A = L->Args;
    N = R->Arity;

    /*  If literal marked inactive, ignore  */

    if ( A[0] ) return CheckRHS(C+1);

    /*  Record types of unbound variables  */

    ForEach(i, 1, N)
    {
	V = A[i];
	if ( ! Variable[V]->TypeRef ) Variable[V]->TypeRef = R->TypeRef[i];
    }

    /* Check for missing values */

    if ( MissingValue(R, A, Value) ) return false;

    /*  Adjust ordering information for recursive literals if required  */

    if ( RecordArgOrders && R == Target )
    {
	ForEach(V, 1, N)
	{
	    W = A[V];

	    if ( Value[V] == UNBOUND || Value[W] == UNBOUND )
	    {
		ThisOrder = '#';
	    }
	    else
	    {
		if ( Variable[V]->TypeRef->Continuous )
		{
		    VVal = FP(Value[V]);
		    WVal = FP(Value[W]);
		}
		else
		{
		    VVal = Variable[V]->TypeRef->CollSeq[Value[V]];
		    WVal = Variable[V]->TypeRef->CollSeq[Value[W]];
		}

		ThisOrder = ( VVal < WVal ? '<' :
			      VVal > WVal ? '>' : '=' );
	    }

	    PrevOrder = L->ArgOrders[V];
	    if ( PrevOrder != ThisOrder )
	    {
		L->ArgOrders[V] = ( ! PrevOrder ? ThisOrder : '#' );
	    }

	    ThisOrder = L->ArgOrders[V];
	    SomeOrder |= ( ThisOrder == '<' || ThisOrder == '>' );
	}

	if ( ! SomeOrder )
	{
	    /*return*/ RecordArgOrders = false;
	}
    }

    /*  Various possible cases  */

    if ( Predefined(R) )
    {
	if ( HasConst(R) )
	{
	    GetParam(&A[2], &KVal);
	}
	else
	{
	    KVal = A[2];
	}

	if ( L->Sign )
	{
	    BoundValue = 0;

	    if ( Satisfies((int)R->Pos, A[1], KVal, Value) )
	    {
		if ( BoundValue )
		{
		    NowBound = true;
		    Value[Target->Arity] = BoundValue;
		}

		if ( CheckRHS(C+1) ) return true;
	    }

	    /*  Restore bound variable if necessary  */

	    if ( NowBound ) Value[Target->Arity] = UNBOUND;

	    return false;
	}
	else
	{
	    return ! Satisfies((int)R->Pos, A[1], KVal, Value) &&
		   CheckRHS(C+1);
	}
    }

    if ( ! L->Sign )
    {
	return ( ! Join(R->Pos, R->PosIndex, A, Value, N, true) ) &&
	         CheckRHS(C+1);
    }

    if ( ! Join(R->Pos, R->PosIndex, A, Value, N, false) ) return false;

    /*  Each tuple found represents a possible binding for the free variables
	in A.  Copy them (to prevent overwriting on subsequent calls to Join)
	and try them in sequence  */

    Bindings = Alloc(NFound+1, Tuple);
    memcpy(Bindings, Found, (NFound+1)*sizeof(Tuple));

    CopyValue = Alloc(MAXVARS+1, Const);
    memcpy(CopyValue, Value, (HighestVar+1)*sizeof(Const));

    Scan = Bindings;

    /*  See whether this literal binds the function value  */

    if ( Value[Target->Arity] == UNBOUND )
    {
	for ( i = 1 ; i <= N && ! FnVar ; i++ )
	{
	    if ( L->Args[i] == Target->Arity ) FnVar = i;
	}
    }

    /* Check rest of RHS */

    while ( ! Stop && (Case = *Scan++) )
    {
	if ( LastSuccess && Case[FnVar] == LastSuccess[FnVar] ) continue;

	ForEach(i, 1, N)
	{
	    V = L->Args[i];
	    if ( Value[V] == UNBOUND ) Value[V] = Case[i];
	}

	if ( CheckRHS(C+1) )
	{
	    if ( FnVar )
	    {
		if ( LastSuccess && Case[FnVar] != LastSuccess[FnVar] )
		{
		    MultipleValues = Stop = true;
		}
		LastSuccess = Case;
	    }
	    else
	    {
		Stop = true;
	    }
	}

	if ( ! Stop ) memcpy(Value, CopyValue, (HighestVar+1)*sizeof(Const));
    }

    if ( LastSuccess && ! Stop )
    {
	/*  Restore values for success  */

	ForEach(i, 1, N)
        {
            V = L->Args[i];
            if ( Value[V] == UNBOUND ) Value[V] = LastSuccess[i];
        }
    }

    pfree(Bindings);
    pfree(CopyValue);
    return ( Stop || LastSuccess );
}



Boolean  Interpret(Relation R, Tuple Case)
/*       ---------  */
{
    int i;

    Target = R;

    for ( i = 0 ; R->Def[i] ; i++ )
    {
	InitialiseValues(Case, R->Arity);

	if ( CheckRHS(R->Def[i]) ) return true;
    }

    return false;
}



void  InitialiseValues(Tuple Case, int N)
/*    ----------------  */
{
    int i;

    ForEach(i, 1, Target->Arity)
    {
	Variable[i]->TypeRef = Target->TypeRef[i];
    }

    ForEach(i, Target->Arity+1, MAXVARS)
    {
	Variable[i]->TypeRef = Nil;
    }

    if ( ! Value )
    {
	Value = Alloc(MAXVARS+1, Const);
    }

    ForEach(i, 0, N-1)
    {
	Value[i] = Case[i];
    }

    ForEach(i, N, MAXVARS)
    {
	Value[i] = UNBOUND;
    }
}
@//E*O*F Release2/Src/interpret.c//
chmod u=rw,g=r,o=r Release2/Src/interpret.c
 
echo x - Release2/Src/join.c
sed 's/^@//' > "Release2/Src/join.c" <<'@//E*O*F Release2/Src/join.c//'
#include  "defns.i"
#include  "extern.i"


	/*  Given tuples T with index TIX, find the tuples that satisfy
	    the column and same-value constraints on case C.  Leave the
	    tuples in Found with their number in NFound  */

	/*  NB: Foil spends a large proportion of its execution time in
	    this single routine.  For that reason, the code has been
	    written for speed (on a DECstation), even though this has
	    reduced its clarity.  If using different hardware, it would
	    probably be worth rewriting this  */

int FoundSize = 0;


Boolean  Join(Tuple *T, Index TIX, Var *A, Tuple C, int N, Boolean YesOrNo)
/*       ----  */
{
    static int	*Pair,			/* Pair[i], Pair[i+1] 
					   are paired variables */
		*Contin,		/* Contin[i] is a continuous variable
					   that must have the given value */
		**Next;			/* Next[i] = next tuple obeying 
					   ith constraint */
    static Boolean *Checked;

    int		*MaxPair,		/* highest pair of same variables */
		*PairPtr,
		*MaxContin,		/* highest continuous variable */
		*ContinPtr,
		MaxCol=0,		/* highest column constraint */
    		MaxNo=0,		/* index numbers in relation */
		**NextPtr,
		**LastNext,
		V, Val, i, j;

    Boolean NoCols;
    Tuple Candidate;

    /*  Allocate arrays first time through  */

    if ( ! FoundSize )
    {
	Pair = Alloc(2*(MAXVARS+1), int);
	Next = Alloc(MAXVARS+1, int *);
	Checked = Alloc(MAXVARS+1, Boolean);

	Contin = Alloc(MAXVARS+1, int);

	FoundSize = 20000;
	Found = Alloc(FoundSize+1, Tuple);
    }

    MaxPair   = Pair;
    MaxContin = Contin;

    NFound = 0;
    OutOfWorld = false;

    /*  Set the column constraints and find pairs of free variables that
	must be the same  */

    memset(Checked+1, 0, N);

    ForEach(i, 1, N)
    {
	/*  If this variable is bound, record a constraint; otherwise see if
	    it is the same as another unbound variable  */

	if ( (V = A[i]) <= Current.MaxVar && (Val = C[V]) != UNBOUND )
	{
	    if ( Variable[V]->TypeRef->Continuous ) *MaxContin++ = i;
	    else
	    if ( ! (Next[MaxCol++] = TIX[i][Val]) ) return false;
	}
	else
	if ( ! Checked[i] )
	{
	    ForEach(j, i+1, N)
	    {
		if ( A[j] == V )
		{
		    *MaxPair++ = i;
		    *MaxPair++ = j;
		    Checked[j] = true;
		}
	    }
	}
    }
    NoCols = MaxCol-- <= 0;
    LastNext = Next + MaxCol;

    while ( true )
    {
	/*  Advance all columns to MaxNo  */

	for ( NextPtr = Next ; NextPtr <= LastNext ; )
	{
	    while ( **NextPtr < MaxNo ) (*NextPtr)++;

	    MaxNo = **NextPtr++;
	}

	if ( MaxNo == FINISH || NoCols && ! T[MaxNo] )
	{
	    Found[NFound] = Nil;
	    return (NFound > 0);
	}
	else
	if ( NoCols || MaxNo == *Next[0] )
	{
	    /*  Found one possibility  -- check same variable constraints  */

	    Candidate = T[MaxNo];

	    for ( PairPtr = Pair ;
		  PairPtr < MaxPair 
                  && Candidate[*PairPtr] == Candidate[*(PairPtr+1)] ;
		  PairPtr += 2 )
		;

	    for ( ContinPtr = Contin ;
		  ContinPtr < MaxContin
		  && Candidate[*ContinPtr] == C[ A[*ContinPtr] ] ;
		  ContinPtr++ )
		;

	    if ( PairPtr >= MaxPair && ContinPtr >= MaxContin )
	    {
		if ( YesOrNo ) return true;

		if ( NFound >= FoundSize )
		{
		    FoundSize += 20000;
		    Realloc(Found, FoundSize+1, Tuple);
		}

		Found[NFound++] = Candidate;

		/*  Check for falling off the end of the (closed) world  */

		for ( i = 1 ; ! OutOfWorld && i <= N ; i++ )
		{
		    if ( Candidate[i] == OUT_OF_RANGE )
		    {
			OutOfWorld = true;
			Found[NFound] = Nil;
			return true;
		    }
		}
	    }

	    MaxNo++;
	}
    }
}
@//E*O*F Release2/Src/join.c//
chmod u=rw,g=r,o=r Release2/Src/join.c
 
echo x - Release2/Src/literal.c
sed 's/^@//' > "Release2/Src/literal.c" <<'@//E*O*F Release2/Src/literal.c//'
/******************************************************************************/
/*									      */
/*	Examine the space of possible literals on a relation		      */
/*									      */
/******************************************************************************/


#include "defns.i"
#include "extern.i"

#define		MAXTIME		100

Var		*Arg = Nil;	/* potential arguments */
Boolean		CountOnly;
float		StartTime;	/* CPU time at start of literal */
int		Ticks,		/* calls of TryArgs() */
                TicksCheck;     /* point to check time again */


	/*  Examine possible variable assignments for next literal using
	    relation R.  If CountOnly specified, this will only set the
	    number of them in R->NTrialArgs; otherwise, it will evaluate
	    the possible arguments  */

void  ExploreArgs(Relation R, Boolean CountFlag)
/*    -----------  */
{
    int	MaxArgs=1;
    Var	V;

    CountOnly = CountFlag;
    if ( CountOnly )
    {
	R->NTrialArgs = 0;
    }
    else
    {
	R->NTried = 0;
    }

    if ( R == Target && ! AnyPartialOrder ||
	 R == GTCONST && Undetermined(Current.Tuples[0]) ) return;

    if ( Predefined(R) )
    {
	if ( R == EQVAR )
	{
	    ExploreEQVAR();
	}
	else
	if ( R == EQCONST )
	{
	    ExploreEQCONST();
	}
	else
	if ( R == GTVAR )
	{
	    ExploreGTVAR();
	}
	else
	if ( R == GTCONST )
	{
	    ExploreGTCONST();
	}

	return;
    }

    if ( CountOnly )
    {
	/*  Carry out a preliminary feasibility check  */

	for ( V = 1 ; MaxArgs <= 1E7 && V <= R->Arity ; V++ )
	{
	    MaxArgs *= EstimatePossibleArgs(R->Type[V]);
	}

	if ( MaxArgs > 1E7 )
	{
	    Verbose(2)
	    {
		printf("\t\t\t\t[%s: too many possibilities]\n", R->Name);
	    }
	    R->NTrialArgs = 0;
	    return;
	}
    }
    else
    {
	Ticks      = 0;
	TicksCheck = 10;
	StartTime  = CPUTime();
    }

    TryArgs(R, 1, Current.MaxVar, 0, 0, 0, true, false);
}



int  EstimatePossibleArgs(int TNo)
/*   --------------------  */
{
    int Sum=1, V;

    ForEach(V, 1, Current.MaxVar)
    {
	if ( Compatible[Variable[V]->Type][TNo] ) Sum++;
    }

    return Sum;
}



	/*  Determine whether a key is acceptable for the relation being
	    explored.  Note that keys are packed bit strings with a 1
	    wherever there is an unbound variable  */


Boolean  AcceptableKey(Relation R, int Key)
/*	 -------------  */
{
    int i;

    if ( ! R->NKeys ) return true;

    ForEach(i, 0, R->NKeys-1)
    {
	if ( (R->Key[i] | Key) == R->Key[i] ) return true;
    }

    return false;
}



	/*  See whether a potential literal is actually a repeat of a literal
	    already in the clause (with perhaps different free variables)  */


Boolean  Repetitious(Relation R, Var *A)
/*       -----------  */
{
    Literal	L;
    Var		V, MaxV;
    int		i, a, N;

    MaxV = Target->Arity;

    ForEach(i, 0, NLit-1)
    {
	L = NewClause[i];
	if ( L->Rel == R &&
	     SameArgs(R->Arity, A, Current.MaxVar, L->Args, MaxV, i+1) )
	{
	    return true;
	}

	if ( L->Sign )
	{
	    N = L->Rel->Arity;
	    ForEach(a, 1, N)
	    {
		if ( (V = L->Args[a]) > MaxV ) MaxV = V;
	    }
	}
    }

    return false;
}



Boolean  Mentioned(Var V, int First)
/*       ---------  */
{
    int		i, a, N;
    Literal	L;

    ForEach(i, First, NLit-1)
    {
	L = NewClause[i];
	N = L->Rel->Arity;

	ForEach(a, 1, N)
	{
	    if ( L->Args[a] == V ) return true;
	}
    }

    return false;
}



	/*  Check whether two aruments are identical up to substitution
	    of free variables  */


Boolean  SameArgs(int N, Var *A1, int MV1, Var *A2, int MV2, int LN)
/*       --------  */
{
    int a;

    ForEach(a, 1, N)
    {
	if ( ( A1[a] <= MV1 ? A2[a] != A1[a] : 
			  ( A2[a]-MV2 != A1[a]-MV1 || Mentioned(A2[a], LN) ) ) )
	{
	    return false;
	}
    }

    return true;
}



	/*  Find arguments for predefined relations  */


void  ExploreEQVAR()
/*    ------------  */
{
    Var		V, W;

    ForEach(V, 1, Current.MaxVar-1)
    {
	if ( Barred[V] ) continue;

	Arg[1] = V;

	ForEach(W, V+1, Current.MaxVar)
	{
	    if ( Barred[W] ) continue;

	    if ( Compatible[Variable[V]->Type][Variable[W]->Type] )
	    {
		if ( CountOnly )
		{
		    EQVAR->NTrialArgs++;
		}
		else
		{
		    Arg[2] = W;
		    EvaluateLiteral(EQVAR, Arg, EQVAR->Bits, Nil);
		    EQVAR->NTried++;
		}
	    }
	}
    }
}

		

void  ExploreEQCONST()
/*    --------------  */
{
    Var		V;
    int		T, i, n;
    Const	C;
    Literal	L;

    /*  Find variables that have bound values  */

    memset(VarUsed, false, Current.MaxVar+1);
    ForEach(i, 0, NLit-1)
    {
	L = NewClause[i];
	if ( L->Rel == EQCONST && L->Sign )
	{
	    VarUsed[L->Args[1]] = true;
	}
    }

    ForEach(V, 1, Current.MaxVar)
    {
	if ( Barred[V] || VarUsed[V] ) continue;

	T = Variable[V]->Type;

	if ( n = Type[T]->NTheoryConsts )
	{
	    Arg[1] = V;

	    ForEach(i, 0, n-1)
	    {
		if ( CountOnly )
		{
		    EQCONST->NTrialArgs++;
		}
		else
		{
		    C = Type[T]->TheoryConst[i];
		    SaveParam(&Arg[2], &C);

		    EvaluateLiteral(EQCONST, Arg, EQCONST->Bits, Nil);
		    EQCONST->NTried++;
		}
	    }
	}
    }
}



void  ExploreGTVAR()
/*    ------------  */
{
    Var		V, W;

    ForEach(V, 1, Current.MaxVar-1)
    {
	if ( Barred[V] || ! Variable[V]->TypeRef->Continuous ) continue;

	Arg[1] = V;

	ForEach(W, V+1, Current.MaxVar)
	{
	    if ( Barred[W] || Variable[W]->Type != Variable[V]->Type ) continue;

	    if ( CountOnly )
	    {
		GTVAR->NTrialArgs++;
	    }
	    else
	    {
		Arg[2] = W;
		EvaluateLiteral(GTVAR, Arg, GTVAR->Bits, Nil);
		GTVAR->NTried++;
	    }
	}
    }
}

		

void  ExploreGTCONST()
/*    --------------  */
{
    Var		V;
    float	Z=MISSING_FP;

    ForEach(V, 1, Current.MaxVar)
    {
	if ( Barred[V] || ! Variable[V]->TypeRef->Continuous ) continue;

	if ( CountOnly )
	{
	    GTCONST->NTrialArgs++;
	}
	else
	{
	    Arg[1] = V;
	    SaveParam(&Arg[2], &Z);

	    EvaluateLiteral(GTCONST, Arg, GTCONST->Bits, Nil);
	    GTCONST->NTried++;
	}
    }
}



    /*  Generate argument lists starting from position This.
	In the partial argument list Arg[1]..Arg[This-1],
	    HiVar is the highest variable (min value MaxVar)
	    FreeVars is the number of free variable occurrences
	    MaxDepth is the highest depth of a bound variable
	    Key gives the key so far
	TryMostGeneral is true when we need to fill all remaining argument
	positions with new free variables.
	RecOK is true when a more general argument list has been found to
	satisfy RecursiveCallOK(), so this must also.

	Return false if hit time limit.  */
	
	
Boolean  TryArgs(Relation R, int This, int HiVar, int FreeVars, int MaxDepth,
	      int Key, Boolean TryMostGeneral, Boolean RecOK)
/*       -------  */
{
    Var		V, W, MaxV;
    Boolean	Prune, UselessSameVar, BoundVar=false;
    int		NewFreeVars, NewMaxDepth, NewKey;
    float       TimeSpent;


    /*  This version contains a time cutout to prevent too much effort
	begin invested in a single literal.  Unfortunately, direct
	monitoring of time is too expensive (in system time and overhead)
	so a more circuitous method that calls CPUTime() rarely is used  */

    if ( ! CountOnly )
    {
        if ( Ticks > TicksCheck )
	{
	    if ( (TimeSpent = CPUTime() - StartTime) > MAXTIME )
	    {
		return false;
	    }
	    else
	    if ( TimeSpent > 0.001 * MAXTIME )
	    {
		TicksCheck += 0.01 * Ticks * MAXTIME / TimeSpent;
	    }
	    else
	    {
		TicksCheck *= 10;
	    }
	}

	Ticks++;
    }

    /*  Check for bound variables  */

    ForEach(V, 1, This-1)
    {
	W = Arg[V];
	BoundVar |= ( W <= Current.MaxVar && Current.Tuples[0][W] != UNBOUND );
    }

    /*  Try with all remaining positions (if any) as new free variables  */

    NewFreeVars = R->Arity - This + 1;

    if ( TryMostGeneral &&
	 HiVar + NewFreeVars <= MAXVARS	/* enough variables */ &&
	 BoundVar 			/* at least one bound */ &&
	 ( ! NewFreeVars || MaxDepth < MAXVARDEPTH ) )
    {
	NewKey = Key;
	ForEach(V, This, R->Arity)
	{
	    Arg[V] = W = HiVar + (V - This + 1);
	    NewKey |= 1<<V;

	    Variable[W]->Type    = R->Type[V];
	    Variable[W]->TypeRef = R->TypeRef[V];
	}

	if ( AcceptableKey(R, NewKey) &&
	     ( R != Target ||
	       RecOK ||
	       (RecOK = RecursiveCallOK(Arg)) ) )
	{
	    if ( CountOnly )
	    {
		R->NTrialArgs++;
	    }
	    else
	    if ( ! Repetitious(R, Arg) )
	    {
		EvaluateLiteral(R, Arg, R->Bits, &Prune);
		R->NTried++;

		if ( Prune && NewFreeVars )
		{
		    Verbose(3) printf("\t\t\t\t(pruning subsumed arguments)\n");
		    return true;
		}
	    }
	}
    }
		
    if ( This > R->Arity ) return true;

    /*  Now try substitutions recursively  */

    MaxV = ( Predefined(R) ? HiVar :
	     This < R->Arity && HiVar < MAXVARS ? HiVar+1 : HiVar );

    for ( V = 1 ; V <= MaxV && BestLitGain < MaxPossibleGain ; V++ )
    {
	if ( V <= Current.MaxVar )
	{
	    if ( Barred[V] ) continue;

	    NewMaxDepth = Max(MaxDepth, Variable[V]->Depth);
	    NewFreeVars = FreeVars;
	}
	else
	{
	    NewMaxDepth = MaxDepth;
	    NewFreeVars = FreeVars+1;
	}

	NewKey = Key;
	if ( V > HiVar )
	{
	    Variable[V]->Type    = R->Type[This];
	    Variable[V]->TypeRef = R->TypeRef[This];
	    NewKey |= 1<<This;
	    if ( ! AcceptableKey(R, NewKey) ) break;
	}
	else
	if ( V <= Current.MaxVar && Current.Tuples[0][V] == UNBOUND )
	{
	    NewKey |= 1<<This;
            if ( ! AcceptableKey(R, NewKey) ) continue;
        }

	/*  Check same variables where impossible  */

	UselessSameVar = false;
	ForEach(W, 1, This-1)
	{
	    UselessSameVar |= Arg[W] == V && R->ArgNotEq[ ArgPair(This,W) ];
	}

	if ( UselessSameVar ||
	     V <= HiVar && ! Compatible[Variable[V]->Type][R->Type[This]] ||
	     NewMaxDepth + (NewFreeVars > 0) > MAXVARDEPTH ||
	     R->BinSym && This == 2 && V < Arg[1] )
	{
	    continue;
	}

	Arg[This] = V;

	if ( ! TryArgs(R, This+1, Max(HiVar, V), NewFreeVars, NewMaxDepth,
		       NewKey, V <= HiVar, RecOK) ) return false;
    }

    return true;
}
@//E*O*F Release2/Src/literal.c//
chmod u=rw,g=r,o=r Release2/Src/literal.c
 
echo x - Release2/Src/main.c
sed 's/^@//' > "Release2/Src/main.c" <<'@//E*O*F Release2/Src/main.c//'
#include  "defns.i"
#include  "extern.i"


	/*  Read parameters and initialise variables  */


void  main(int Argc, char *Argv[])
/*    ----  */
{
    int		o, i, Cases, Errors;
    extern char	*optarg;
    extern int	optind;
    Boolean	FirstTime=true;
    Var		V;
    extern Var	*Arg;	/* in literal.c */
    Relation	R;
    Tuple	Case;
    char	Line[200];
    Const	Predicted;

    /* Check overlaying of Const and float */

    if ( sizeof(Const) != sizeof(float) )
    {
	printf("Integers and floating point numbers are different sizes\n");
	printf("Alter declaration of type Const (defns.i) and recompile\n");
	exit(1);
    }

    printf("FFOIL 2.0   [December 1995]\n---------\n");

    /*  Process options  */

    while ( (o = getopt(Argc, Argv, "nNusa:f:g:V:d:A:w:l:t:m:v:")) != EOF )
    {
	if ( FirstTime )
	{
	    printf("\n    Options:\n");
	    FirstTime = false;
	}

	switch (o)
	{
	    case 'n':	NEGLITERALS = NEGEQUALS = false;
			printf("\tno negated literals\n");
			break;

	    case 'N':	NEGLITERALS = false;
			printf("\tnegated literals only for =, > relations\n");
			break;

	    case 's':	SIMPLIFY = false;
			printf("\tno global pruning of clauses\n");
			break;

	    case 'u':   /*UNIFORMCODING = true;*/
	                printf("\tuniform coding of literals not available\n");
	                break;

	    case 'a':	MINACCURACY = atof(optarg);
			printf("\tminimum clause accuracy %g%%\n",MINACCURACY);
			MINACCURACY /= 100;
			break;

	    case 'f':	MINALTFRAC = atof(optarg);
			printf("\tbacked-up literals have %g%% of best gain\n",
				MINALTFRAC);
			MINALTFRAC /= 100;
			break;

	    case 'g':	i = atoi(optarg);
			printf("\tuse determinate literals when gain <= ");
	                printf("%d%% possible\n", i);
			DETERMINATE = i / 100.0;
			break;

	    case 'V':   MAXVARS = atoi(optarg);
			if ( MAXVARS > pow(2.0, 8*sizeof(Var)-1.0) )
			{
			    MAXVARS = pow(2.0, 8*sizeof(Var)-1.0) -0.9;
			}
                        printf("\tallow up to %d variables\n", MAXVARS);
	                break;

	    case 'd':	MAXVARDEPTH = atoi(optarg);
			printf("\tmaximum variable depth %d\n", MAXVARDEPTH);
			break;

	    case 'w':   MAXWEAKLITS = atoi(optarg);
                        printf("\tallow up to %d consecutive weak literals\n",
				MAXWEAKLITS);
	                break;

	    case 'l':	MAXPOSSLIT = atoi(optarg)+1;
			printf("\tmaximum %d backups from one literal\n", 
				MAXPOSSLIT-1);
			break;

	    case 't':	MAXALTS = atoi(optarg);
			printf("\tmaximum %d total backups\n", MAXALTS);
			break;

	    case 'm':   MAXTUPLES = atoi(optarg);
                        printf("\tmaximum %d tuples \n", MAXTUPLES);
	                break;

	    case 'v':	VERBOSITY = atoi(optarg);
			printf("\tverbosity level %d\n", VERBOSITY);
			break;

	    case '?':	printf("unrecognised option\n");
			exit(1);
	}
    }

    /*  Set up predefined relations.

	These are treated specially in Join().  Rather than giving explicit
	pos tuples, the Pos field identifies the relation  */

    /*  Note: EQCONST and GTCONST take one argument and one parameter
	(a theory constant or floating-point threshold).  To simplify the
	code for all other relations, this parameter is packed into a
	"standard" arglist; the number of variable positions that it
	occupies will depend on the implementation.  */

    Reln = Alloc(10, Relation);

    EQVAR = AllocZero(1, struct _rel_rec);

    EQVAR->Name = "=";
    EQVAR->Arity = 2;

    EQVAR->Type = AllocZero(3, int);
    EQVAR->TypeRef = AllocZero(3, TypeInfo);

    EQVAR->Pos = (Tuple *) 0;
    EQVAR->BinSym = true;

    EQCONST = AllocZero(1, struct _rel_rec);

    EQCONST->Name = "==";
    EQCONST->Arity = 1;

    EQCONST->Type = AllocZero(2, int);
    EQCONST->TypeRef = AllocZero(2, TypeInfo);

    EQCONST->Pos = (Tuple *) 1;
    EQCONST->BinSym = false;

    GTVAR = AllocZero(1, struct _rel_rec);

    GTVAR->Name = ">";
    GTVAR->Arity = 2;

    GTVAR->Type = AllocZero(3, int);
    GTVAR->TypeRef = AllocZero(3, TypeInfo);

    GTVAR->Pos = (Tuple *) 2;
    GTVAR->BinSym = true;

    GTCONST = AllocZero(1, struct _rel_rec);

    GTCONST->Name = ">";
    GTCONST->Arity = 1;

    GTCONST->Type = AllocZero(2, int);
    GTCONST->TypeRef = AllocZero(2, TypeInfo);

    GTCONST->Pos = (Tuple *) 3;
    GTCONST->BinSym = false;

    MaxRel = 3;

    ForEach(i, 0, MaxRel)
    {
	Reln[i]->PossibleTarget = false;
    }

    /*  Read input  */

    ReadTypes();

    CheckTypeCompatibility();

    ReadRelations();

    /*  Initialise all global variables that depend on parameters  */

    Variable    = Alloc(MAXVARS+1, VarInfo);
    DefaultVars = Alloc(MAXVARS+1, Var);

    ForEach(V, 0, MAXVARS)
    {
	Variable[V] = Alloc(1, struct _var_rec);

	if ( V == 0 )
	{
	    Variable[0]->Name = "_";
	}
	else
	if ( V <= 26 )
	{
	    Variable[V]->Name = Alloc(2, char);
	    Variable[V]->Name[0] = 'A' + V - 1;
	    Variable[V]->Name[1] = '\0';
	}
	else
	{
	    Variable[V]->Name = Alloc(3, char);
	    Variable[V]->Name[0] = 'A' + ((V-27) / 26);
	    Variable[V]->Name[1] = 'A' + ((V-27) % 26);
	    Variable[V]->Name[2] = '\0';
	}

	DefaultVars[V] = V;
    }

    Barred = AllocZero(MAXVARS+1, Boolean);

    VarUsed = Alloc(MAXVARS+1, Boolean);

    ToBeTried = Alloc(MAXALTS+1, Alternative);
    ForEach(i, 0, MAXALTS)
    {
	ToBeTried[i] = Alloc(1, struct _backup_rec);
	ToBeTried[i]->UpToHere = Nil;
    }

    Scheme = Alloc(MAXARGS+1, int);

    PartialOrder = Alloc(MAXARGS+1, Boolean *);
    ForEach(V, 1, MAXARGS)
    {
	PartialOrder[V] = Alloc(MAXVARS+1, Boolean);
    }

    Possible = Alloc(MAXPOSSLIT+1, PossibleLiteral);
    ForEach(i, 1, MAXPOSSLIT)
    {
	Possible[i] = Alloc(1, struct _poss_lit_rec);
	Possible[i]->Args = Alloc(MAXARGS+1, Var);
    }

    Arg = Alloc(MAXARGS+1, Var);
    Arg[0] = 0;	/* active */

    /*  Allocate space for trial recursive call */

    RecursiveLitOrders = Alloc(1, Ordering *);
    RecursiveLitOrders[0] = Alloc(MAXARGS+1, Ordering);

    /*  Find plausible orderings  for constants of each type  */

    OrderConstants();

    /* Find Definitions */

    ForEach(i, 0, MaxRel)
    {
	R = RelnOrder[i];

	if ( R->PossibleTarget )
	{
	    FindDefinition(R);
	}
    }

    /*  Test definitions  */

    while ( gets(Line) )
    {
	R = Nil;
	for ( i = 0 ; i <= MaxRel && ! R ; i++ )
	{
	    if ( ! strcmp(RelnOrder[i]->Name, Line) ) R = RelnOrder[i];
	}

	if ( ! R )
	{
	    printf("\nUnknown function %s\n", Line);
	    exit(1);
	}
	else
	{
	    printf("\nTest function %s\n", Line);
	}

	Cases = Errors = 0;
	Current.MaxVar = HighestVarInDefinition(R);

	while ( Case = ReadTuple(R) )
	{
	    Cases++;
	    Predicted = UNBOUND;

	    if ( ! Interpret(R, Case) ||
		 (Predicted=Value[R->Arity]) != Case[R->Arity] )
	    {
		Verbose(1)
		{
		    printf("    (%s) ",
			   Predicted == UNBOUND ? "?" : ConstName[Predicted]);
		    PrintTuple(Case, R->Arity, R->TypeRef, false);
		}
		Errors++;
	    }
	}

	printf("Summary: %d error%s in %d trial%s\n",
		Errors, Plural(Errors), Cases, Plural(Cases));
    }

    exit(0);
}
@//E*O*F Release2/Src/main.c//
chmod u=rw,g=r,o=r Release2/Src/main.c
 
echo x - Release2/Src/order.c
sed 's/^@//' > "Release2/Src/order.c" <<'@//E*O*F Release2/Src/order.c//'
/******************************************************************************/
/*									      */
/*	Routines for controlling recursive definitions.  The basic idea	      */
/*	is to keep track of partial orders between variables (using	      */
/*	either predefined or discovered constant orders) and to ensure	      */
/*	that there is an ordering of all recursive literals that will	      */
/*	guarantee termination.  See Cameron-Jones and Quinlan, IJCAI'93	      */
/*									      */
/******************************************************************************/


#include "defns.i"
#include "extern.i"


    /*  Examine relationships among variables: LHSVar <,=,>,#  anyvar and
	anyvar = anyvar  */

void  ExamineVariableRelationships()
/*    ---------------------------- */
{
    Var		V, W;
    Const	VVal, WVal;
    Ordering	X, ThisX;
    Tuple	*Scan, Case;
    int		*TypeOrder;
    Boolean	FirstTime=true;

    /*  First reset all partial orders  */

    ForEach(V, 1, Target->Arity)
    {
	memset(PartialOrder[V], '#', Current.MaxVar+1);
	PartialOrder[V][V] = '=';
    }

    if ( ! Current.NTot ) return;

    ForEach(V, 1, Current.MaxVar-1)
    {
	if ( ! Variable[V]->TypeRef ||
	     Variable[V]->TypeRef->Continuous ||
	     Current.Tuples[0][V] == UNBOUND ) continue;

	if ( V <= Target->Arity )
	{
	    TypeOrder = Target->TypeRef[V]->CollSeq;
	}

	ForEach(W, V+1, Current.MaxVar)
	{
	    if ( ! Variable[W]->TypeRef ||
	         Variable[W]->TypeRef->Continuous ||
		 Current.Tuples[0][W] == UNBOUND ||
		 ! Compatible[Variable[V]->Type][Variable[W]->Type] )  continue;

	    for ( X = 0, Scan = Current.Tuples ; X != '#' && *Scan ; Scan++ )
	    {
		Case = *Scan;

		if ( (VVal = Case[V]) == (WVal = Case[W]) )
		{
		    ThisX = '=';
		}
		else
		if ( V > Target->Arity ||
		     ! Variable[V]->TypeRef->Ordered ||
		     ! Variable[W]->TypeRef->Ordered )
		{
		    ThisX = '#';
		}
		else
		{
		    ThisX = ( TypeOrder[VVal] < TypeOrder[WVal] ? '<' : '>' );
		}

		if ( ! X )
		{
		    X = ThisX;
		}
		else
		if ( X != ThisX )
		{
		    X = '#';
		}
	    }

	    if ( X != '#' )
	    {
		Verbose(2)
		{
		    if ( FirstTime )
		    {
			printf("\t\tNote");
			FirstTime = false;
		    }
		    printf(" %s%c%s", Variable[V]->Name, X, Variable[W]->Name);
		}
	    }

	    if ( X == '=' ) Barred[W] = true;

	    /*  Record partial order for recursive literals.  If polarity
		is fixed, treat > as #  */

	    if ( V <= Target->Arity && X != '#' )
	    {
		ThisX = PartialOrder[V][W] = X;
		AnyPartialOrder |= ThisX == '<' || ThisX == '>';

		if ( W <= Target->Arity )
		{
		    ThisX = PartialOrder[W][V] =
			X == '<' ? '>' : X == '>' ? '<' : '=';
		    AnyPartialOrder |= ThisX == '<' || ThisX == '>';
		}
	    }
	}
    }
    Verbose(2) putchar('\n');
}



	/*  Vet proposed arguments for recursive literal.
	    Uses a mapping from ThisOrder x Cell to ThisOrder  */


Boolean  RecursiveCallOK(Var *A)
/*       ---------------  */
{
    int			i, j, k, N, NRowLeft, Count, NRow, BestCount, BestCol;
    Ordering		*ThisCall, ThisOrder, BestOrder, Cell;
    Boolean		SomeInequality=false;

    static Ordering	**Map=Nil;
    static Boolean	*ColLeft, *RowLeft;

    if ( A && ! AnyPartialOrder ) return false;

    if ( ! Map )
    {
	N = (int) '>';	/* max of '#', '<', '>', '=' */

	Map = Alloc(N+1, Ordering *);
	
	/*  We want the final value for a column to be
		'=' if it contains only '=' entries
		'<' if it contains only '<' and/or '=' entries
		'>' if it contains only '>' and/or '=' entries
	    and '#' otherwise  */

	Map['='] = Alloc(N+1, Ordering);
	Map['<'] = Alloc(N+1, Ordering);
	Map['>'] = Alloc(N+1, Ordering);
	Map['#'] = Alloc(N+1, Ordering);

	Map['=']['='] = '=';
	Map['=']['<'] = '<';
	Map['=']['>'] = '>';
	Map['=']['#'] = '#';

	Map['<']['='] = '<';
	Map['<']['<'] = '<';
	Map['<']['>'] = '#';
	Map['<']['#'] = '#';

	Map['>']['='] = '>';
	Map['>']['<'] = '#';
	Map['>']['>'] = '>';
	Map['>']['#'] = '#';

	Map['#']['='] = '#';
	Map['#']['<'] = '#';
	Map['#']['>'] = '#';
	Map['#']['#'] = '#';

	ColLeft = Alloc(MAXARGS+1, Boolean);
	RowLeft = Alloc(1001, Boolean);	/* assume <= 1000 recursive lits! */
    }

    N = Target->Arity;

    memset(ColLeft, true, N+1);

    NRow = NRecLitDef + NRecLitClause;
    memset(RowLeft, true, NRow+1);

    /*  First need to establish ordering constraints for these arguments.
	(Skip this if A is nil, called from pruning routines)  */

    if ( A )
    {
	NRowLeft = NRow + 1;
	ThisCall = RecursiveLitOrders[0];
	ForEach(i, 1, N)
	{
	    if ( A[i] > Current.MaxVar )
	    {
		ThisCall[i] = '#';
	    }
	    else
	    {
		ThisCall[i] = PartialOrder[i][A[i]];
		SomeInequality |= ThisCall[i] == '<' || ThisCall[i] == '>';
	    }
	}

	if ( ! SomeInequality ) return false;
    }
    else
    {
	NRowLeft = NRow;
	RowLeft[0] = false;
    }

    /*  Check for a possible ordering by
	* finding a column that has only (< or >) and = orders
	* delete rows containing (< or >)
	* continue until no rows remain  */

    /*  This routine is also invoked during the pruning phase, when some
	literals have been (perhaps temporarily) removed from the most
	recent clause.  Their orderings are marked as inactive; these are
	treated as if already covered  */

    ForEach(j, NRecLitDef+1, NRow)
    {
	if ( RecursiveLitOrders[j][0] )
	{
	    RowLeft[j] = false;
	    NRowLeft--;
	}
    }

    NScheme = 0;
    while( NRowLeft > 0 )
    {
        BestCol = BestCount = 0;

	ForEach(k, 1, N)
	{
	    if ( ! ColLeft[k] ) continue;

	    Count = 0;
	    ThisOrder = ( Target->TypeRef[k]->FixedPolarity ? '>' : '=' );

	    for ( j = 0 ; ThisOrder != '#' && j <= NRow ; j++ )
	    {
	        if ( ! RowLeft[j] ) continue;

		Cell = RecursiveLitOrders[j][k];
		if ( Cell != '=' ) Count++;

		ThisOrder = Map[ThisOrder][Cell];
	    }

	    if ( ThisOrder != '#' && Count > BestCount )
	    {
	        BestCount = Count;
		BestCol = k;
		BestOrder = ThisOrder;
	    }
	}

	if ( ! BestCol )
	{
	    /*  Recursive call not OK  */

	    return false;
	}

	/*  Process best column  */

	ForEach(j, 0, NRow)
	{
	    if ( RowLeft[j] && RecursiveLitOrders[j][BestCol] != '=' )
	    {
	        RowLeft[j] = false;
		NRowLeft--;
	    }
        }

	ColLeft[BestCol] = false;

	Scheme[NScheme++] = ( BestOrder == '>' ? BestCol : -BestCol );
    }

    return  NRowLeft == 0;
}



	/*  Add argument order information for recursive literal.
	    Note: this must be performed before calling NewState so that
	    new variables are correctly given ordering #  */


void  AddOrders(Literal L)
/*    ---------  */
{
    Var		V, W;

    /*  Allocate ordering and mark as active  */

    if ( ! L->ArgOrders )
    {
	L->ArgOrders = Alloc(Target->Arity+1, Ordering);
	L->ArgOrders[0] = 0;
    }

    ForEach(V, 1, Target->Arity)
    {
	W = L->Args[V];
	L->ArgOrders[V] = ( W <= Current.MaxVar ? PartialOrder[V][W] : '#' );
    }
}



	/*  Keep track of argument orders for recursive literals.
	    (The first cell is reserved for testing)  */

void  NoteRecursiveLit(Literal L)
/*    ----------------  */
{
    static int	RecLitSize=0;
    int		i;

    NRecLitClause++;
    i = NRecLitDef + NRecLitClause;

    if ( i >= RecLitSize )
    {
	RecLitSize += 100;
	Realloc(RecursiveLitOrders, RecLitSize, Ordering *);
    }

    RecursiveLitOrders[i] = L->ArgOrders;
}
@//E*O*F Release2/Src/order.c//
chmod u=rw,g=r,o=r Release2/Src/order.c
 
echo x - Release2/Src/output.c
sed 's/^@//' > "Release2/Src/output.c" <<'@//E*O*F Release2/Src/output.c//'
/******************************************************************************/
/*									      */
/*	All output routines						      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"


void  PrintTuple(Tuple C, int N, TypeInfo *TypeRef, Boolean ShowPosNeg)
/*    ----------  */
{
    int		i;

    printf("\t");

    ForEach(i, 1, N)
    {
        if ( TypeRef ? TypeRef[i]->Continuous :
		       Variable[i]->TypeRef && 
		       Variable[i]->TypeRef->Continuous )
	{
	    /*  Continuous value  */

	    if ( FP(C[i]) == MISSING_FP )
	    {
		printf("?");
	    }
	    else
	    {
		printf("%g", FP(C[i]));
	    }
	}
	else
	{
	    printf("%s", ( C[i] == UNBOUND ? "?" : ConstName[C[i]]) );
	}

	if ( i < N ) putchar(',');
    }

    if ( ShowPosNeg )
    {
	printf(": %c", ( Undetermined(C) ? '.' : Positive(C) ? '+' : '-') );
    }

    putchar('\n');
}


    
void  PrintTuples(Tuple *TT, int N)
/*    -----------  */
{
    while ( *TT )
    {
	PrintTuple(*TT, N, Nil, true);
	TT++;
    }
}



void  PrintSpecialLiteral(Relation R, Boolean RSign, Var *A)
/*    -------------------  */
{
    Const	ThConst;
    float	Thresh;

    if ( R == EQVAR )
    {
        printf("%s%s%s", Variable[A[1]]->Name, RSign ? "=":"<>",
                         Variable[A[2]]->Name);
    }
    else
    if ( R == EQCONST )
    {
	GetParam(&A[2], &ThConst);

	printf("%s%s%s", Variable[A[1]]->Name, RSign ? "=" : "<>",
			 ConstName[ThConst]);
    }
    else
    if ( R == GTVAR )
    {
        printf("%s%s%s", Variable[A[1]]->Name, RSign ? ">" : "<=",
                         Variable[A[2]]->Name);
    }
    else
    if ( R == GTCONST )
    {
	GetParam(&A[2], &Thresh);

	if ( Thresh == MISSING_FP )
	{
	    printf("%s%s", Variable[A[1]]->Name, RSign ? ">" : "<=");
	}
	else
	{
	    printf("%s%s%g", Variable[A[1]]->Name, RSign ? ">" : "<=", Thresh);
	}
    }
}



void  PrintComposedLiteral(Relation R, Boolean RSign, Var *A)
/*    --------------------  */
{
    int i, V;

    if ( Predefined(R) )
    {
	PrintSpecialLiteral(R, RSign, A);
    }
    else
    {
	if ( ! RSign )
	{
	    printf("not(");
	}

	printf("%s", R->Name);
	ForEach(i, 1, R->Arity)
	{
	    printf("%c", (i > 1 ? ',' : '('));
	    if ( (V = A[i]) <= MAXVARS )
	    {
		printf("%s", Variable[V]->Name);
	    }
	    else
	    {
		printf("_%d", V - MAXVARS);
	    }
	}
	putchar(')');

	if ( ! RSign )
	{
	    putchar(')');
	}
    }
}



void  PrintLiteral(Literal L)
/*    ------------  */
{
    PrintComposedLiteral(L->Rel, L->Sign, L->Args);
}



void  PrintClause(Relation R, Clause C, Boolean Cut)
/*    -----------  */
{
    int		Lit;

    PrintComposedLiteral(R, true, DefaultVars);

    for ( Lit = 0 ; C[Lit] ; Lit++ )
    {
	if( C[Lit]->Args[0] ) continue;

	printf("%s ", ( Lit ? "," : " :-" ));

	PrintLiteral(C[Lit]);
    }
    printf("%s.\n", ( Cut ? ", !" : "" ));
}



	/*  Print clause, substituting for variables equivalent to constants  */

void  PrintSimplifiedClause(Relation R, Clause C, Boolean Cut)
/*    ---------------------  */
{
    int		i, Lit, NextUnbound;
    Literal	L;
    Var		V, Bound, *SaveArgs, *Unbound;
    char	**HoldVarNames;
    Const	TC;
    Boolean	NeedComma=false;

    SaveArgs = Alloc(MAXVARS+1, Var);
    Unbound = Alloc(MAXVARS+1, Var);

    /*  Save copy of variable names  */

    HoldVarNames = Alloc(MAXVARS+1, char *);
    ForEach(V, 1, MAXVARS)
    {
	HoldVarNames[V] = Variable[V]->Name;
    }

    /*  Substitute for equal variables  */

    for ( Lit = 0 ; L = C[Lit] ; Lit++ )
    {
        if ( L->Rel == EQVAR && L->Sign )
	{
	    Substitute(Variable[L->Args[2]]->Name, Variable[L->Args[1]]->Name);
	}
    }

    /*  Set up alternate names for variables equated to theory constants  */

    for ( Lit = 0 ; L = C[Lit] ; Lit++ )
    {
        if ( L->Rel == EQCONST && L->Sign )
	{
	    GetParam(&L->Args[2], &TC);
	    Substitute(Variable[L->Args[1]]->Name, ConstName[TC]);
	}
    }

    PrintComposedLiteral(R, true, DefaultVars);
    Bound = R->Arity;

    for ( Lit = 0 ; L = C[Lit] ; Lit++ )
    {
	/*  Save literal args  */

	memcpy(SaveArgs, L->Args, (L->Rel->Arity+1) * sizeof(Var));

	/*  Update bound vars and change unbound vars in negated literals  */

	NextUnbound = MAXVARS;
	memset(Unbound, 0, MAXVARS+1);

	ForEach(i, 1, L->Rel->Arity )
	{
	    if ( (V = L->Args[i]) > Bound )
	    {
		if ( ! L->Sign )
		{
		    if ( ! Unbound[V] ) Unbound[V] = ++NextUnbound;
		    L->Args[i] = Unbound[V];
		}
		else
		{
		    Bound = V;
		}
	    }
	}
	
	/*  Skip literals that are implicit in changed names  */

        if ( L->Rel != EQCONST && L->Rel != EQVAR || ! L->Sign )
	{
	    printf("%s", ( NeedComma ? ", " : " :- ") );

	    PrintLiteral(L);
	    NeedComma = true;
	}

	/*  Restore args  */

	memcpy(L->Args, SaveArgs, (L->Rel->Arity+1) * sizeof(Var));
    }
    printf("%s.\n", ( ! Cut ? "" : NeedComma ? ", !" : " :- !") );

    ForEach(V, 1, MAXVARS)
    {
	Variable[V]->Name = HoldVarNames[V];
    }

    pfree(SaveArgs);
    pfree(Unbound);
    pfree(HoldVarNames);
}



void  Substitute(char *Old, char *New)
/*    ----------  */
{
    Var V;

    ForEach(V, 1, MAXVARS)
    {
	if ( Variable[V]->Name == Old ) Variable[V]->Name = New;
    }
}



void  PrintDefinition(Relation R)
/*    ---------------  */
{
    int		Cl;
    Clause	C;

    putchar('\n');
    for ( Cl = 0 ; C=R->Def[Cl] ; Cl++ )
    {
	PrintSimplifiedClause(R, C, R->Def[Cl+1] != 0);
    }

    printf("\nTime %.1f secs\n", CPUTime());
}
@//E*O*F Release2/Src/output.c//
chmod u=rw,g=r,o=r Release2/Src/output.c
 
echo x - Release2/Src/prune.c
sed 's/^@//' > "Release2/Src/prune.c" <<'@//E*O*F Release2/Src/prune.c//'
/******************************************************************************/
/*									      */
/*	Stuff for pruning clauses and definitions.  Clauses are first	      */
/*	`quickpruned' to remove determinate literals that introduce unused    */
/*	variables; if this causes the definition to become less accurate,     */
/*	these are restored.  Then literals are removed one at a time to	      */
/*	see whether they contribute anything.  A similar process is	      */
/*	followed when pruning definitions: clauses are removed one at a	      */
/*	time to see whether the remaining clauses suffice		      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"

extern Boolean RecordArgOrders,	/* for interpret.c */
	       MultipleValues;

Boolean	       RecursiveDef=false;

#define  MarkLiteral(L,X)	{ (L)->Args[0]=X;\
				  if ((L)->ArgOrders) (L)->ArgOrders[0]=X;}


	/*  See whether literals can be dropped from a clause  */

void  PruneNewClause()
/*    --------------  */
{
    int		Cover, Errs, i, j, k;
    Boolean	Changed=false;
    Literal	L, LL;
    Var		V;

    Cover = Current.NOrigTot;
    Errs = Current.NOrigTot - Current.NOrigPos;

    Verbose(2)
    {
	printf("\nInitial clause (%d errs): ", Errs);
	PrintClause(Target, NewClause, true);
    }

    /*  Promote any literals of form A=B or B=c to immediately after B
	appears in the clause  */

    ForEach(i, 0, NLit-1)
    {
	L = NewClause[i];

	if ( L->Sign && ( L->Rel == EQVAR || L->Rel == EQCONST ) )
	{
	    V = ( L->Rel == EQVAR ? L->Args[2] : L->Args[1] );

	    if ( V > Target->Arity )
	    {
		ForEach(j, 1, i)
		{
		    LL = NewClause[j-1];
		    if ( LL->Sign && Contains(LL->Args, LL->Rel->Arity, V) )
		    {
			break;
		    }
		}
	    }
	    else
	    {
		j = 0;
	    }

	    for ( k = i ; k > j ; k-- )
	    {
		NewClause[k] = NewClause[k-1];
	    }

	    NewClause[j] = L;
	}
    }

    /*  Look for unexpected relations V=W or V=c  */

    CheckVariables();

    /*  Check for quick pruning of apparently useless literals  */

    if ( QuickPrune(NewClause, Target->Arity, false) )
    {
	if ( SatisfactoryNewClause(Cover, Errs) )
	{
	    Verbose(/*3*/2) printf("\tAccepting quickpruned clause\n");
	    Changed = true;
	}
	else
	{
	    /*  Mark all literals as active again  */

	    ForEach(i, 0, NLit-1)
	    {
		MarkLiteral(NewClause[i], 0);
	    }
	}
    }

    Changed |= RedundantLiterals(Errs);

    /*  Remove unnecessary literals from NewClause and reconstruct
	all information on clause  */

    if ( Changed )
    {
	Cleanup(NewClause);
    }

    RecoverState(NewClause, false);
}



	/*  Look for unexpected equivalences of variables or variables
	    with unchanging values.  If found, insert pseudo-literals
	    into the clause so that redundant literals can be pruned  */

void  CheckVariables()
/*    --------------  */
{
    Var A, B;

    ForEach(A, 1, Current.MaxVar)
    {
	if ( TheoryConstant(Current.Tuples[0][A], Variable[A]->TypeRef) &&
	     ConstantVar(A, Current.Tuples[0][A]) )
	{
	    if ( ! Known(EQCONST, A, 0) )
	    {
		Insert(A, EQCONST, A, Current.Tuples[0][A]);
	    }
	}
	else
	{
	    ForEach(B, A+1, Current.MaxVar)
	    {
		if ( Compatible[Variable[A]->Type][Variable[B]->Type] &&
		     IdenticalVars(A,B) && ! Known(EQVAR, A, B) )
		{
		    Insert(B, EQVAR, A, B);
		}
	    }
	}
    }
}



	/*  See whether a constant is a theory constant  */

Boolean  TheoryConstant(Const C, TypeInfo T)
/*       --------------  */
{
    int i;

    if ( T->Continuous ) return false;

    ForEach(i, 0, T->NTheoryConsts-1)
    {
	if ( C == T->TheoryConst[i] ) return true;
    }

    return false;
}



	/*  Check for variable with constant value  */

Boolean  ConstantVar(Var V, Const C)
/*       -----------  */
{
    Tuple	Case, *TSP;

    for ( TSP = Current.Tuples ; Case = *TSP++; )
    {
	if ( Positive(Case) && Case[V] != C ) return false;
    }

    return true;
}



	/*  Check for identical variables  */

Boolean  IdenticalVars(Var V, Var W)
/*       -------------  */
{
    Tuple	Case, *TSP;

    for ( TSP = Current.Tuples ; Case = *TSP++; )
    {
	if ( Positive(Case) &&
	     ( Unknown(V, Case) || Case[V] != Case[W] ) ) return false;
    }

    return true;
}



	/*  Make sure potential literal isn't already in clause  */

Boolean  Known(Relation R, Var V, Var W)
/*       -----  */
{
    int 	i;
    Literal	L;

    ForEach(i, 0, NLit-1)
    {
	L = NewClause[i];
	if ( L->Rel == R && L->Sign &&
	     L->Args[1] == V && ( ! W || L->Args[2] == W ) )
	{
	    return true;
	}
    }

    return false;
}



	/*  Insert new literal after V first bound  */

void  Insert(Var V, Relation R, Var A1, Const A2)
/*    ------  */
{
    int 	i=0, j;
    Literal	L;

    if ( V > Target->Arity )
    {
	ForEach(i, 1, NLit)
	{
	    L = NewClause[i-1];
	    if ( L->Sign && Contains(L->Args, L->Rel->Arity, V) ) break;
	}
    }

    /*  Insert literal before NewClause[i]  */

    if ( ++NLit % 100 == 0 ) Realloc(NewClause, NLit+100, Literal);
    for ( j = NLit ; j > i ; j-- )
    {
	NewClause[j] = NewClause[j-1];
    }

    L = AllocZero(1, struct _lit_rec);

    L->Rel  = R;
    L->Sign = true;
    L->Bits = 0;

    if ( R == EQVAR )
    {
	L->Args = AllocZero(3, Var);
	L->Args[1] = A1;
	L->Args[2] = A2;
    }
    else
    {
	L->Args = AllocZero(2+sizeof(Const), Var);
	L->Args[1] = A1;
	SaveParam(L->Args+2, &A2);
    }

    NewClause[i] = L;
    Verbose(2)
    {
	printf("\tInsert literal ");
	PrintSpecialLiteral(R, true, L->Args);
	printf("\n");
    }
}



	/*  Check for variable in argument list  */

Boolean  Contains(Var *A, int N, Var V)
/*       --------  */
{
    int i;

    ForEach(i, 1, N)
    {
	if ( A[i] == V ) return true;
    }

    return false;
}


		
	/*  Remove determinate literals that introduce useless new variables
	    (that are not used by any subsequent literal)  */

Boolean  QuickPrune(Clause C, Var MaxBound, Boolean ValueBound)
/*       ----------  */
{
    Var		V, W, NewMaxBound;
    Literal	L;
    Boolean	Retain=false, SomeUsed=true, Changed=false;

    if ( (L = C[0]) == Nil )
    {
	memset(VarUsed, false, MaxBound+1);
	return false;
    }

    L->Args[0] = 0;

    NewMaxBound = MaxBound;
    if ( L->Sign )
    {
	ForEach(V, 1, L->Rel->Arity)
	{
	    if ( (W = L->Args[V]) > NewMaxBound ) NewMaxBound = L->Args[V];
	    else
	    Retain |= W == Target->Arity && ! ValueBound ;
	}
    }

    Changed = QuickPrune(C+1, NewMaxBound, ValueBound || Retain);

    if ( ! Retain && L->Sign == 2 )
    {
	SomeUsed = false;
	ForEach(V, MaxBound+1, NewMaxBound)
	{
	    SomeUsed = VarUsed[V];
	    if ( SomeUsed ) break;
	}
    }

    if ( ! Retain && ! SomeUsed && C[1] )
    {
	/*  Mark this literal as inactive  */

	MarkLiteral(L, 1);

	Verbose(/*3*/2)
	{
	    printf("\tQuickPrune literal ");
	    PrintLiteral(L);
	    putchar('\n');
	}

	Changed = true;
    }
    else
    ForEach(V, 1, L->Rel->Arity)
    {
	VarUsed[L->Args[V]] = true;
    }

    return Changed;
}



Boolean  SatisfactoryNewClause(int Cover, int Errs)
/*       ---------------------  */
{
    Boolean	RecursiveLits=false;
    Literal	L;
    int		i, ErrsNow=0;
    Tuple	Case;

    /*  Prepare for redetermining argument orders  */

    ForEach(i, 0, NLit-1)
    {
	if ( (L = NewClause[i])->Args[0] ) continue;

	if ( L->Rel == Target )
	{
	    RecursiveLits = true;
	    memset(L->ArgOrders, 0, Target->Arity+1);
	}
    }

    RecordArgOrders = RecursiveLits;

    for ( i = 0 ; i < StartClause.NTot ; i++ )
    {
	Case = StartClause.Tuples[i];
	InitialiseValues(Case, Target->Arity);

	if ( CheckRHS(NewClause) &&
	     Value[Target->Arity] != FnValue[Case[0]] )
	{
	    if ( ++ErrsNow > Errs )
	    {
		RestoreTypeRefs();
		return false;
	    }
	}
    }
    RestoreTypeRefs();

    /*  Also check that recursion still well-behaved  */

    RecoverState(NewClause, false);
    /*CheckOriginalCaseCover();*/

    return Current.NOrigTot >= Cover &&
	   Current.NOrigTot - Current.NOrigPos <= Errs &&
	   ( RecursiveLits ? RecordArgOrders && RecursiveCallOK(Nil) : true );
}


    /*  Remove any deleted literals from a clause  */

void  Cleanup(Clause C)
/*    -------  */
{
    int		i, j;
    Literal	L;
    Boolean	None=true;

    for ( i = 0 ; L = C[i] ; )
    {
	if ( L->Args[0] )
	{
	    None = false;
	    FreeLiteral(L);

	    for ( j = i ; C[j] ; j++ )
	    {
		C[j] = C[j+1];
	    }
	}
	else
	{
	   i++;
	   RecursiveDef |= ( L->Rel == Target );
	}
    }

    if ( ! None ) RenameVariables(C);
}



    /*  Rename variables in a clause  */

void  RenameVariables(Clause C)
/*    ---------------  */
{
    Var		*NewVar, Next, SaveNext, V;
    int		l, i;
    Literal	L;

    NewVar = AllocZero(MAXVARS+1, Var);
    Next = Target->Arity+1;

    for ( l = 0 ; L = C[l] ; l++ )
    {
	if ( L->Args[0] ) continue;

	SaveNext = Next;
	ForEach(i, 1, L->Rel->Arity)
	{
	    V = L->Args[i];

	    if ( V > Target->Arity )
	    {
		if ( ! NewVar[V] ) NewVar[V] = Next++;

		L->Args[i] = NewVar[V];
	    }
	}

	/*  New variables appearing in negated lits are still free  */

	if ( ! L->Sign )
	{
	    Next = SaveNext;
	    ForEach(V, 1, MAXVARS)
	    if ( NewVar[V] >= Next ) NewVar[V] = 0;
	}
    }

    pfree(NewVar);
}



	/*  Omit the first unnecessary literal.
	    This version prunes from the end of the clause; if a literal
	    can't be dropped when it is examined, it will always be
	    needed, since dropping earlier literals can only increase
	    the number of minus tuples getting through to this literal  */


Boolean  RedundantLiterals(int ErrsNow)
/*       -----------------  */
{
    int		Cover, i, j;
    Boolean	Changed=false;
    Literal	L;

    /*  Check for the latest literal, omitting which would not increase
	the number of errors.  Note: checking last literal is reinstated
	since clause may contain errors  */

    Cover = Current.NOrigTot;

    for ( i = NLit-1 ; i >= 0 ; i-- )
    {
	L = NewClause[i];

	/*  Can't omit a literal that is needed to bind a variable appearing in
	    a later negated literal  */

        if ( L->Args[0] || EssentialBinding(i) )
	{
	    continue;
	}

	MarkLiteral(L, 1);

	if ( SatisfactoryNewClause(Cover, ErrsNow) )
	{
	    Verbose(/*3*/2)
	    {
		printf("\t\t");
		PrintLiteral(L);
		printf(" removed\n");
	    }
	    Cover = Current.NOrigTot;
	    Changed = true;
	}
	else
	{
	    Verbose(/*3*/2)
	    {
		printf("\t\t");
		PrintLiteral(L);
		printf(" essential\n");
	    }
	    MarkLiteral(L, 0);

	    /*  If this literal is V=W, where W is a non-continuous variable
		bound on the RHS of the clause, substitute for W throughout
		and remove the literal  */

	    if ( L->Rel == EQVAR && L->Sign && L->Args[2] > Target->Arity &&
		 ! Variable[L->Args[2]]->TypeRef->Continuous )
	    {
		ForEach(j, i, NLit-1)
		{
		    NewClause[j] = NewClause[j+1];
		}
		NLit--;
		ReplaceVariable(L->Args[2], L->Args[1]);
	    }
	}
    }

    return Changed;
}



	/*  Can't omit a literal that is needed to bind a variable appearing in
	    a later negated literal relation, or whose omission would leave a
	    later literal containing all new variables  */

#define  NONE	-1
#define  MANY	1000000

Boolean  EssentialBinding(int LitNo)
/*       ----------------  */
{
    int		i, j, b, *UniqueBinding;
    Boolean	Needed=false, Other;
    Literal	L;
    Var		V, NArgs;

    /*  UniqueBinding[V] = NONE (if V not yet bound)
			 = i    (if V bound only by ith literal)
			 = MANY (if V bound by more than one literal)  */

    UniqueBinding = Alloc(MAXVARS+1, int);
    ForEach(V, 1, MAXVARS)
    {
	UniqueBinding[V] = ( V < Target->Arity ? MANY : NONE );
    }

    for ( i = 0 ; i < NLit && ! Needed ; i++ )
    {
	if ( (L = NewClause[i])->Args[0] ) continue;

	NArgs = L->Rel->Arity;

	if ( i > LitNo )
	{
	    if ( Predefined(L->Rel) || ! L->Sign )
	    {
		ForEach(j, 1, NArgs)
		{
		    Needed |= UniqueBinding[L->Args[j]] == LitNo;
		}
	    }
	    else
	    {
		Other = false;
		ForEach(j, 1, NArgs)
		{
		    b = UniqueBinding[L->Args[j]];
		    Other |= b != NONE && b != LitNo;
		}
		Needed = ! Other;
	    }

	    if ( Needed )
	    {
		Verbose(/*3*/2)
		{
		    printf("\t\t");
		    PrintLiteral(NewClause[LitNo]);
		    printf(" needed for binding ");
		    PrintLiteral(NewClause[i]);
		    putchar('\n');
		}
	    }
	}

	if ( L->Sign )
	{
	    /*  Update binding records for new variables  */

	    ForEach(j, 1, L->Rel->Arity)
	    {
		V = L->Args[j];

		if ( UniqueBinding[V] == NONE )
		{
		    UniqueBinding[V] = i;
		}
		else
		if ( UniqueBinding[V] != i )
		{
		    UniqueBinding[V] = MANY;
		}
	    }
	}
    }

    pfree(UniqueBinding);
    return Needed;
}




	/*  Substitute for variable in clause  */

void  ReplaceVariable(Var Old, Var New)
/*    ---------------  */
{
    int 	i, a;
    Literal	L;
    Boolean	Bound=false;

    ForEach(i, 0, NLit-1)
    {
	L = NewClause[i];

	if ( L->Sign || Bound )
	{
	    ForEach(a, 1, L->Rel->Arity)
	    {
		if ( L->Args[a] == Old )
		{
		    L->Args[a] = New;
		    Bound = true;
		}
	    }
	}
    }
}



	/*  Global clause pruning: try to discard literals without
	    increasing the total number of errors   */

void  SimplifyClauses()
/*    ---------------  */
{
    int		i, j, Errs, Best, LastMod=0, Mod;
    Clause	C;
    Literal	L, BestL;

    Current.MaxVar = HighestVarInDefinition(Target);

    Verbose(2) printf("\nSimplify clauses");

    RecordArgOrders = false;

    while ( true )
    {
	Best = CountErrs(LastMod, true);
	BestL = Nil;

	Verbose(2)
	    printf(" (%d errors)\n", Best);

	ForEach(i, 0, NCl-2)
	{
	    Verbose(3) printf("\nClause %d\n", i);
	    C = Target->Def[i];

	    for ( j = 0 ; L = C[j] ; j++ )
	    {
		if ( L->Args[0] ) continue;

		MarkLiteral(L, 1);
		Errs = CountErrs(i, false);
		MarkLiteral(L, 0);

		Verbose(3)
		{
		    printf("  literal ");
		    PrintLiteral(L);
		    if ( Errs > StartDef.NTot )
		    {
			printf(" required for recursion control\n");
		    }
		    else
		    {
			printf(" -> %d\n", Errs);
		    }
		}

		if ( Errs < Best || ! BestL && Errs == Best )
		{
		    Best = Errs;
		    BestL = L;
		    Mod = i;
		}
	    }
	}

	if ( ! BestL ) break;
	MarkLiteral(BestL, 1);
	LastMod = Mod;

	Verbose(2)
	{
	    printf("  remove ");
	    PrintLiteral(BestL);
	    printf(" from clause %d", Mod);
	}
    }

    ForEach(i, 0, NCl-2)
    {
	Cleanup(Target->Def[i]);
    }
}



int CountErrs(int LastMod, Boolean Change)
/*  ---------  */
{
    static int  *CoveredBy=Nil;
    static Boolean *Correct;
    Boolean	All=false, OK;
    int		i, j, CC, Wrong=0;
    Clause	C;

    if ( ! CoveredBy )
    {
	CoveredBy = (int *) calloc(StartDef.NTot, sizeof(int));
	Correct = (Boolean *) malloc(StartDef.NTot);
	All = true;
    }

    for ( i = 0 ; i < StartDef.NTot ; i++ )
    {
	if ( CoveredBy[i] < LastMod )
	{
	    CC = CoveredBy[i];
	    OK = Correct[i];
	}
	else
	{
	    CC = -1;
	    OK = false;

	    for ( j = LastMod ; j < NCl && CC < 0 ; j++ )
	    {
		InitialiseValues(StartDef.Tuples[i], Target->Arity);
		MultipleValues = false;

		if ( CheckRHS(Target->Def[j]) )
		{
		    CC = j;
		    OK = ! MultipleValues && Value[Target->Arity] == FnValue[i];

		    if ( RecursiveDef && UnsoundRecursion(Target->Def[j]) )
		    {
			return StartDef.NTot+1;
		    }
		}
		else
		if ( ! All && j == LastMod && j != CoveredBy[i] )
		{
		    CC = CoveredBy[i];
		    OK = Correct[i];
		}
	    }
	}

	if ( ! OK ) Wrong++;

	if ( Change )
	{
	    CoveredBy[i] = ( CC >= 0 ? CC : NCl );
	    Correct[i]   = OK;
	}
    }

    return Wrong;
}



	/*  Check whether simplified clauses can be discarded  */

void  SiftClauses()
/*    -----------  */
{
    int		i, j, Covers, Last, Remove, Retain=0, *Change;
    Boolean	*Delete, *Needed, Correct[2], Alter=false;
    Clause	C;

    if ( ! NCl ) return;

    Verbose(3) printf("\nSifting clauses\n");

    Change = Alloc(NCl, int);
    Delete = AllocZero(NCl, Boolean);
    Needed = AllocZero(NCl, Boolean);

    Current.MaxVar = HighestVarInDefinition(Target);

    /*  Now examine the possibility of deleting a clause  */

    while ( true )
    {
	memset(Needed, false, NCl);

	ForEach(i, 0, NCl-1)
	{
	    Change[i] = 0;
	}

	for ( i = 0 ; i < StartDef.NTot ; i++ )
	{
	    Covers = Correct[1] = 0;

	    for ( j = 0 ; j < NCl-1 && Covers <= 1 ; j++ )
	    {
		if ( Delete[j] ) continue;

		InitialiseValues(StartDef.Tuples[i], Target->Arity);

		if ( CheckRHS(Target->Def[j]) )
		{
		    if ( Covers == 0 )
		    {
			Last = j;
		    }
		    else
		    if ( RecursiveDef && UnsoundRecursion(Target->Def[j]) )
		    {
			Value[Target->Arity] = UNBOUND;

			/*  Previous clause cannot be removed because it
			    establishes a variable ordering essential for
			    guaranteeing recursive termination  */

			if ( ! Needed[Last] )
			{
			    Verbose(3)
			      printf("\tClause %d protects clause %d\n", Last, j);
			}
			Needed[Last] = true;
		    }

		    Correct[Covers++] = ( Value[Target->Arity] == FnValue[i] );
		}
	    }

	    if ( Covers )
	    {
		Change[Last] += Correct[1] - Correct[0];
	    }
	}

	/*  Remove the most useless clause  */

	Remove = NCl;

	ForEach(i, 0, NCl-2)
	{
	    if ( Needed[i] || Delete[i] ) continue;

	    if ( Change[i] >= 0 &&
		 ( Remove == NCl || Change[i] > Change[Remove] ) )
	    {
		Remove = i;
	    }
	}

	if ( Remove == NCl ) break;

	Verbose(3)
	    printf("  remove clause %d (benefit %d)\n", Remove, Change[Remove]);
	Delete[Remove] = Alter = true;
    }


    if ( Alter )
    {
	Verbose(2) printf("\nDelete clauses\n  ");

	Alter = false;
	ForEach(i, 0, NCl-1)
	{
	    if ( Delete[i] )
	    {
		Verbose(2)
		{
		    printf("%s%d", ( Alter ? ", " : "" ), i);
		    Alter = true;
		}
		FreeClause(Target->Def[i]);
	    }
	    else
	    {
		Target->Def[Retain++] = Target->Def[i];
	    }
	}
	Target->Def[NCl = Retain] = Nil;
	Verbose(2) printf("\n");
    }

    pfree(Change);
    pfree(Delete);
    pfree(Needed);
}



Boolean UnsoundRecursion(Clause C)
/*      ----------------  */
{
    Literal	L;
    int		i, j, *TypeOrder;
    Var		V;
    Const	Head, Body;
    Boolean	OK, GT;

    for ( i = 0 ; (L = C[i]) ; i++ )
    {
	if ( L->Rel != Target ) continue;

	for ( OK = false, j = 0 ; ! OK && j < NScheme ; j++ )
	{
	    V = abs(Scheme[j]);
	    TypeOrder = Target->TypeRef[V]->CollSeq;

	    Head = Value[V];
	    Body = Value[L->Args[V]];

	    if ( Head != Body )
	    {
		GT = TypeOrder[Head] > TypeOrder[Body];
		if ( GT != (Scheme[j] > 0) )
		{
		    return true;
		}
		else
		{
		    OK = true;
		}
	    }
	}
	if ( ! OK ) return true;
    }

    return false;
}




    /*  Find highest variable in any clause  */

Var  HighestVarInDefinition(Relation R)
/*   ----------------------  */
{
    Var		V, HiV;
    Literal	L;
    Clause	C;
    int		i;

    HiV = R->Arity;

    for ( i = 0 ; R->Def[i] ; i++ )
    {
	for ( C = R->Def[i] ; L = *C ; C++ )
	{
	    if ( L->Sign )
	    {
		ForEach(V, 1, L->Rel->Arity)
		{
		    if ( L->Args[V] > HiV ) HiV = L->Args[V];
		}
	    }
	}
    }

    return HiV;
}



void  RestoreTypeRefs()
/*    ---------------  */
{
    Var V;

    ForEach(V, 1, Current.MaxVar)
    {
	Variable[V]->TypeRef = Type[Variable[V]->Type];
    }
}
@//E*O*F Release2/Src/prune.c//
chmod u=rw,g=r,o=r Release2/Src/prune.c
 
echo x - Release2/Src/search.c
sed 's/^@//' > "Release2/Src/search.c" <<'@//E*O*F Release2/Src/search.c//'
/******************************************************************************/
/*									      */
/*	Routines to control search.  The search for clauses is basically      */
/*	greedy, but a limited number of alternative possibilities is	      */
/*	kept on hand							      */
/*									      */
/******************************************************************************/


#include "defns.i"
#include "extern.i"

Boolean	FirstSave;	/* flag for printing */


    /*	At any stage, the MAXPOSSLIT best literals to use next are
	kept in the array Possible.  This procedure puts a new literal
	into the list if it qualifies  */


void  ProposeLiteral(Relation R, Boolean TF, Var *A,
		     int Size, float LitBits, int OPos, int OTot,
		     float Gain, Boolean Weak)
/*    --------------  */
{
    PossibleLiteral Entry;
    int		    i, j, ArgSize;

    /*  See where this literal would go.  Other things being equal, prefer
	an unnegated literal - regarding "<=" as unnegated */

    i = NPossible;
    while ( i > 0 &&
	    ( Gain > Possible[i]->Gain + 1E-6 ||
	      Gain == Possible[i]->Gain && TF && ! Possible[i]->Sign ) )
    {
	i--;
    }

    if ( i >= MAXPOSSLIT )  return;

    if ( NPossible < MAXPOSSLIT ) NPossible++;

    Entry = Possible[MAXPOSSLIT];

    for ( j = MAXPOSSLIT ; j > i+1 ; j-- )
    {
	Possible[j] = Possible[j-1];
    }

    Possible[i+1] = Entry;

    ArgSize = (R->Arity+1)*sizeof(Var);
    if ( HasConst(R) ) ArgSize += sizeof(Const);

    Entry->Gain     = Gain;
    Entry->Rel      = R;
    Entry->Sign     = TF;
    memcpy(Entry->Args, A, ArgSize);
    Entry->Bits	    = LitBits;
    Entry->NewSize  = Size;
    Entry->PosCov   = OPos;
    Entry->TotCov   = OTot;
    Entry->WeakLits = ( Weak ? NWeakLits+1 : 0 );
}



    /*  When all possible literals have been explored, the best of them
	(if there are any) is extracted and used.  Others with gain
	close to the best are entered as backups  */

Literal  SelectLiteral()
/*       -------------  */
{
    int i;

    if ( ! NPossible ) return Nil;

    FirstSave = true;
    ForEach(i, 2, NPossible)
    {
	if ( Possible[i]->Gain >= MINALTFRAC * Possible[1]->Gain )
	{
	    Remember(MakeLiteral(i), Possible[i]->PosCov, Possible[i]->TotCov);
	}
    }

    return MakeLiteral(1);
}



Literal  MakeLiteral(int i)
/*       -----------  */
{
    int		ArgSize;
    Literal	L;

    L = AllocZero(1, struct _lit_rec);

    L->Rel  = Possible[i]->Rel;
    L->Sign = Possible[i]->Sign;
    L->Bits = Possible[i]->Bits;

    L->WeakLits = Possible[i]->WeakLits;

    ArgSize = (L->Rel->Arity+1)*sizeof(Var);
    if ( HasConst(L->Rel) ) ArgSize += sizeof(Const);
    L->Args = Alloc(ArgSize, Var);
    memcpy(L->Args, Possible[i]->Args, ArgSize);

    return L;
}



    /*  The array ToBeTried contains all backup points so far.
	This procedure sees whether another proposed backup will fit
	or will displace an existing backup  */

void  Remember(Literal L, int OPos, int OTot)
/*    --------  */
{
    int		i, j;
    Alternative	Entry;
    float	InfoGain;

    InfoGain = OPos *
	       (Info(StartClause.NPos, StartClause.NTot) - Info(OPos, OTot));

    /*  See where this entry would go  */

    for ( i = NToBeTried ; i && ToBeTried[i]->Value < InfoGain ; i-- )
	;

    if ( i >= MAXALTS )
    {
	FreeLiteral(L);
	return;
    }

    if ( NToBeTried < MAXALTS ) NToBeTried++;

    Entry = ToBeTried[MAXALTS];

    for ( j = MAXALTS ; j > i+1 ; j-- )
    {
	ToBeTried[j] = ToBeTried[j-1];
    }

    ToBeTried[i+1] = Entry;

    if ( Entry->UpToHere ) pfree(Entry->UpToHere);

    Entry->UpToHere = Alloc(NLit+2, Literal);
    memcpy(Entry->UpToHere, NewClause, (NLit+1)*sizeof(Literal));
    Entry->UpToHere[NLit]   = L;
    Entry->UpToHere[NLit+1] = Nil;
    Entry->Value	    = InfoGain;

    Verbose(1)
    {
	if ( FirstSave )
	{
	    putchar('\n');
	    FirstSave = false;
	}
	printf("\tSave ");
	PrintLiteral(L);
	printf(" (%d,%d value %.1f)\n", OPos, OTot, InfoGain);
    }
}



Boolean  Recover()
/*       -------  */
{
    int		i;
    Clause	C;
    Alternative	Entry;

    if ( SavedClause || ! NToBeTried || MAXRECOVERS-- < 1 ) return Nil;

    Entry = ToBeTried[1];
    C = ToBeTried[1]->UpToHere;

    Verbose(1)
    {
	printf("\nRecover to ");
	PrintClause(Target, C, false);
    }

    ForEach(i, 2, NToBeTried)
    {
	ToBeTried[i-1] = ToBeTried[i];
    }
    ToBeTried[NToBeTried] = Entry;
    NToBeTried--;

    RecoverState(C, true);
    /*CheckOriginalCaseCover();*/
    ExamineVariableRelationships();

    return true;
}



void  FreeLiteral(Literal L)
/*    -----------  */
{
    pfree(L->Args);
    if ( L->ArgOrders ) pfree(L->ArgOrders);
    pfree(L);
}



void  FreeClause(Clause C)
/*    ----------  */
{
    Clause CC;

    for ( CC = C ; *CC ; CC++ )
    {
	FreeLiteral(*CC);
    }
    pfree(C);
}
@//E*O*F Release2/Src/search.c//
chmod u=rw,g=r,o=r Release2/Src/search.c
 
echo x - Release2/Src/state.c
sed 's/^@//' > "Release2/Src/state.c" <<'@//E*O*F Release2/Src/state.c//'
/******************************************************************************/
/*									      */
/*	A state, summarising a point in the search for a clause, consists     */
/*	of a set of tuples and various counts.  The routines here set up      */
/*	an initial state for a relation and produce the new state that	      */
/*	results when a literal is added to the current (partial) clause	      */
/*									      */
/******************************************************************************/


#include  "defns.i"
#include  "extern.i"


	/*  Set up original state for search for definition  */


void  OriginalState(Relation R)
/*    -------------  */
{
    int		i, TupleArity, *Freq;
    Tuple	*Scan;
    Const	Best=0;
    Clause	C;
    double	drand48();

    /*  Check that the target relation has at least two arguments and that
	the last (the function value) is of discrete type  */

    if ( R->Arity < 2 || R->TypeRef[R->Arity]->Continuous )
    {
	printf("\n*** Relation %s is not a learnable function\n    ", R->Name);
	printf("(need at least 2 arguments with the last of discrete type)\n");
	exit(1);
    }

    /*  Establish number of variables hence size of tuples, and set variable 
        types equal to the type of variable in that position in the relation,
        and variable depths to zero  */

    StartDef.MaxVar = Current.MaxVar = R->Arity;
    TupleArity = R->Arity+1;

    ForEach(i, 1, R->Arity)
    {
	Variable[i]->Type    = R->Type[i];
	Variable[i]->TypeRef = R->TypeRef[i];
	Variable[i]->Depth   = 0;
    }
    FnRange = R->TypeRef[R->Arity]->NValues;

    StartDef.NPos = 0;
    StartDef.NTot = Number(R->Pos);
    Realloc(FnValue, StartDef.NTot+1, Const);
    AllTuples = StartDef.NTot * FnRange;
    
    if ( StartDef.NTot > MAXTUPLES )
    {
	printf("Training Set Size exceeds tuple limit: ");
	printf("%d > %d\n", StartDef.NTot, MAXTUPLES);
	printf("Rerun with larger MAXTUPLES to proceed further\n");
	exit(0);
    }

    StartDef.Tuples = Alloc(StartDef.NTot+1, Tuple);

    i = 0;

    for ( Scan = R->Pos ; *Scan ; Scan++ )
    {
	StartDef.Tuples[i] = Alloc(TupleArity, Const);
	memcpy(StartDef.Tuples[i], *Scan, TupleArity*sizeof(Const));
	StartDef.Tuples[i][0] = i;
	FnValue[i] = StartDef.Tuples[i][R->Arity];
	StartDef.Tuples[i][R->Arity] = UNBOUND;
	i++;
    }

    /*  Find default value, if any  */

    Freq = AllocZero(MaxConst+1, int);
    ForEach(i, 0, StartDef.NTot-1)
    {
	Freq[FnValue[i]]++;
    }

    ForEach(i, 1, MaxConst)
    {
	if ( ! Best || Freq[i] > Freq[Best] ) Best = i;
    }

    FnDefault = ( Freq[Best] == 1 ? 0 : Best );
    cfree(Freq);

    StartDef.Tuples[StartDef.NTot] = Nil;
    StartDef.NOrigPos = StartDef.NPos;
    StartDef.NOrigTot = StartDef.NTot;

    Realloc(Flags, StartDef.NTot+1, char);

    StartDef.BaseInfo = Info(StartDef.NPos, StartDef.NTot);

    if ( ! LogFact )
    {
	LogFact = Alloc(1001, float);

	LogFact[0] = 0.0;
	ForEach(i, 1, 1000)
	{
	    LogFact[i] = LogFact[i-1] + Log2(i);
	}
    }
}



void  NewState(Literal L, int NewSize)
/*    --------  */
{
    FormNewState(L->Rel, L->Sign, L->Args, NewSize);
    AcceptNewState(L->Rel, L->Args, NewSize);
}



	/*  Construct new state in New  */

void  FormNewState(Relation R, Boolean RSign, Var *A, int NewSize)
/*    ------------  */
{
    Tuple	*TSP, Case, *Bindings, Instance;
    int		i, N, RN;
    Boolean	BuiltIn=false, Match, NotNegated;
    Const	X2;

    if ( Predefined(R) )
    {
	/*  Prepare for calls to Satisfies()  */

	BuiltIn = true;
	RN = (int) R->Pos;
	if ( HasConst(R) )
	{
	    GetParam(&A[2], &X2);
	}
	else
	{
	    X2 = A[2];
	}
    }

    N = R->Arity;

    /*  Find highest variable in new clause  */

    New.MaxVar = Current.MaxVar;
    if ( RSign )
    {
        ForEach(i, 1, N)
	{
	    New.MaxVar = Max(A[i], New.MaxVar);
	}
    }

    New.Tuples = Alloc(NewSize+1, Tuple);

    New.NPos = New.NTot = 0;

    BoundValue = 0;
    for ( TSP = Current.Tuples ; Case = *TSP++ ; )
    {
        if ( MissingValue(R,A,Case) ) continue;

	Match = ( BuiltIn ? Satisfies(RN, A[1], X2, Case) :
	          Join(R->Pos, R->PosIndex, A, Case, N, ! RSign) );
	NotNegated = RSign != 0;
	if ( Match != NotNegated ) continue;

	if ( ! BuiltIn && RSign )
	{
	    /*  Add tuples from R->Pos  */

	    CheckSize(New.NTot, NFound, &NewSize, &New.Tuples);

	    Bindings = Found;
	    while ( Instance = *Bindings++ )
	    {
		New.Tuples[New.NTot] = Extend(Case, Instance, A, N);
		New.NTot++;
	    }
	}
	else
	{
	    CheckSize(New.NTot, 1, &NewSize, &New.Tuples);

	    New.Tuples[New.NTot] = InitialiseNewCase(Case);
	    if ( BoundValue )
	    {
		New.Tuples[New.NTot][Target->Arity] = BoundValue;
	    }
	    New.NTot++;
	}
    }
    New.Tuples[New.NTot] = Nil;

    /*  Make sure all positive tuples come first  */

    ForEach(i, 0, New.NTot-1)
    {
	Case = New.Tuples[i];
	if ( Positive(Case) )
	{
	    if ( i != New.NPos )
	    {
		New.Tuples[i] = New.Tuples[New.NPos];
		New.Tuples[New.NPos] = Case;
	    }
	    New.NPos++;
	}
    }
}


	/*  Move state in New to Current  */

void  AcceptNewState(Relation R, Var *A, int NewSize)
/*    --------------  */
{
    int		i, N, MaxDepth=0;
    Var		V;

    if ( Current.Tuples != StartClause.Tuples )
    {
	FreeTuples(Current.Tuples, true);
    }

    if ( New.MaxVar > Current.MaxVar )
    {
	/*  New variable(s) - update type and depth info  */

        N = R->Arity;
        ForEach(i, 1, N)
	{
            if ( (V = A[i]) <= Current.MaxVar )
	    {
                MaxDepth = Max(MaxDepth, Variable[V]->Depth);
	    }
	}
        MaxDepth++;

        ForEach(i, 1, N)
	{
            if ( (V = A[i]) > Current.MaxVar )
	    {
                Variable[V]->Type    = R->Type[i];
		Variable[V]->TypeRef = R->TypeRef[i];
                Variable[V]->Depth   = MaxDepth;
	    }
	}
    }

    New.BaseInfo = Info(New.NPos, New.NTot);

    /*  Move all information across and resize tuples if necessary  */

    Current = New;

    if ( New.NTot < NewSize )
    {
	Realloc(Current.Tuples, Current.NTot+1, Tuple);
    }

    Current.BaseInfo = Info(Current.NPos, Current.NTot);

    CheckOriginalCaseCover();
}



    /*  Rebuild a training set by applying the literals in a clause
	to the copy of the training set  */

void  RecoverState(Clause C, Boolean MakeNewClause)
/*    ------------  */
{
    int		i, SaveVerbosity;
    Literal	L;
    float	FalseBits;

    /*  Turn off messages during recovery  */

    SaveVerbosity = VERBOSITY;
    VERBOSITY = 0;

    if ( Current.Tuples != StartClause.Tuples )
    {
        FreeTuples(Current.Tuples, true);
	Current = StartClause;
    }

    ForEach(i, 1, Target->Arity)
    {
	memset(PartialOrder[i], '#', MAXVARS+1);
    	PartialOrder[i][i] = '=';
    }
    AnyPartialOrder = false;

    memset(Barred, false, MAXVARS+1);

    NRecLitClause = NDetLits = ClauseBits = 0;

    for ( NLit = 0 ; (L = C[NLit]) ; NLit++ )
    {
	if ( L->Args[0] ) continue;

	if ( MakeNewClause ) NewClause[NLit] = L;

	if ( L->Rel == Target )
	{
	    ExamineVariableRelationships();
	    AddOrders(L);
	}

	NewState(L, Current.NTot);

	if ( L->Sign == 2 )
	{
	    NDetLits++;
	}
	else
	{
	    FalseBits = L->Bits - Log2(NLit+1.001-NDetLits);
	    ClauseBits += Max(0, FalseBits);
	}

	if ( L->Rel == Target ) NoteRecursiveLit(L);
	NWeakLits = L->WeakLits;
    }

    if ( MakeNewClause ) NewClause[NLit] = Nil;
    VERBOSITY = SaveVerbosity;
}



void  CheckSize(int SoFar, int Extra, int *NewSize, Tuple **TSAddr)
/*    ---------  */
{
    if ( SoFar+Extra > *NewSize )
    {
	*NewSize += Max(Extra, 1000);
	Realloc(*TSAddr, *NewSize+1, Tuple);
    }
}



Tuple  InitialiseNewCase(Tuple Case)
/*     -----------------  */
{
    Tuple       NewCase;
    int         i;
    
    NewCase = Alloc(New.MaxVar+1, Const);
    memcpy(NewCase, Case, (Current.MaxVar+1)*sizeof(Const));

    ForEach(i, Current.MaxVar+1, New.MaxVar)
    {
	NewCase[i] = UNBOUND;
    }

    return NewCase;
}



    /*  Tack extra variables on a tuple  */

Tuple  Extend(Tuple Case, Tuple Binding, Var *A, int N)
/*     ------  */
{
    Tuple	NewCase;
    int		i;

    NewCase = InitialiseNewCase(Case);

    ForEach(i, 1, N)
    {
	NewCase[A[i]] = Binding[i];
    }

    return NewCase;
}



void  CheckOriginalCaseCover()
/*    ----------------------  */
{
    Tuple *TSP, Case;

    ClearFlags;

    Current.NOrigTot = 0;
    for ( TSP = Current.Tuples ; Case = *TSP++ ; )
    {
        if ( ! TestFlag(Case[0], TrueBit) )
	{
            SetFlag(Case[0], TrueBit);
	    Current.NOrigTot++;
	}
	if ( ! Positive(Case) ) SetFlag(Case[0], FalseBit);
    }

    Current.NOrigPos = PosDefinite(TrueBit);
}



int  PosDefinite(int Bit)
/*   -----------  */
{
    Tuple *TSP, Case;
    int Count=0;

    for ( TSP = StartDef.Tuples ; Case = *TSP++ ; )
    {
	if ( TestFlag(Case[0], Bit) && ! TestFlag(Case[0], FalseBit) )
	{
	    Count++;
	}
    }

    return Count;
}



    /*  Free up a bunch of tuples  */

void  FreeTuples(Tuple *TT, Boolean TuplesToo)
/*    -------  */
{
    Tuple *P;

    if ( TuplesToo )
    {
	for ( P = TT ; *P ; P++ )
	{
	    pfree(*P);
	}
    }

    pfree(TT);
}



    /*  Find log (base 2) factorials using tabulated values and Stirling's
	approximation (adjusted)  */

double  Log2Fact(int n)
/*      --------  */
{
    return ( n < 1000 ? LogFact[n] :
	     (n+0.5) * Log2(n) - n * Log2e + Log2sqrt2Pi - (n*0.7623)/820000 );
}
@//E*O*F Release2/Src/state.c//
chmod u=rw,g=r,o=r Release2/Src/state.c
 
echo x - Release2/Src/utility.c
sed 's/^@//' > "Release2/Src/utility.c" <<'@//E*O*F Release2/Src/utility.c//'
#include "defns.i"
#include <sys/time.h>
#include <sys/resource.h>

extern int	VERBOSITY;


	/*  Protected memory allocation routines on call for memory which
	    is not allocated rather than let program continue erroneously */

void  *pmalloc(unsigned arg)
/*     -------  */
{
    void *p;

    p = (void *) malloc(arg);
    if( p || !arg ) return p;
    printf("\n*** malloc erroneously returns NULL\n");
    exit(1);
}



void  *prealloc(void * arg1, unsigned arg2)
/*     --------  */
{
    void *p;

    if ( arg1 == Nil ) return pmalloc(arg2);

    p = (void *)realloc(arg1,arg2); 
    if( p || !arg2 ) return p;
    printf("\n*** realloc erroneously returns NULL\n");
    exit(1);
}



void  *pcalloc(unsigned arg1, unsigned arg2)
/*     -------  */
{
    void *p;

    p = (void *)calloc(arg1,arg2);
    if( p || !arg1 || !arg2 ) return p;
    printf("\n*** calloc erroneously returns NULL\n");
    exit(1);
}



void  pfree(void *arg)
{
    if ( arg ) free(arg);
}



float  CPUTime()
{
    struct rusage usage;

    getrusage(RUSAGE_SELF, &usage);

    return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
    	   (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1E6;
}
@//E*O*F Release2/Src/utility.c//
chmod u=rw,g=r,o=r Release2/Src/utility.c
 
exit 0
