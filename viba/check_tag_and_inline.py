"""viba.check_tag_and_inline — a design's tags, and the chains that spread them.

check_tag_and_inline(design, config) -> Result[None]. Ok(None) when every
product the design writes has all its tags different once the inline chains are
spread, and every inline chain ends; Err names the first mistake.

    from viba.check_tag_and_inline import check_tag_and_inline
    check_tag_and_inline(pool)       # -> Ok(None), or Err(the first mistake)

A product's untagged members are inline slots (viba-reflect.md section 4): the
members of the definition written there spread into the product that names it.
Two writing mistakes only become visible after that spreading, and both are
properties of the design alone — no material is involved:

- the same tag twice in one product, inlined members counted: a base's `$x`
  next to a written `$x`, or one base inlined twice;
- an inline chain that comes back to a definition it is already spreading
  (`A = A * $x int`), which has no expansion to read at all.

Both are Err here, before anything judges the design, so the answer does not
depend on what the design is later compared with. `is_sub_type` and `serialize`
refuse them too, but only where their walk happens to reach them: a product
nested in a tagged body or in a container element is read only when the
comparison descends into it. This check walks the design once, from every
definition, and reads every product it can reach.

A tag written twice at one level never gets this far: compiling the file already
refuses it (`PoolAddFile` answers `duplicate member`). What is left for the
inline reading is a tag that only repeats once a base has been spread, and a
chain that never stops unfolding.

The map does the reading (viba.reflect): a product's members come from
``VibaAccess.members_of`` and its chain from ``VibaAccess.inline_cycle``, and
the accessor is built here from a config, so a caller never holds one. The
default is the language's own config (``reflect.language_config``); a caller
with a different vocabulary passes a ``VibaReflectConfig`` of its own. Which
names stand for a unit does not change a tag — a unit has none and never
inlines — but saying what you mean costs nothing.
"""

from __future__ import annotations

from viba.reflect import Config, VibaAccess, language_config
from viba.type import DuplicateTagError, Err, InlineCycleError, Ok, Result
from viba.viba_type_descriptor import (
    EXPONENT,
    PRODUCT,
    SUM,
    TAGGED,
    TUPLE,
    TYPE_APP,
    VibaPool,
)

__all__ = ["check_tag_and_inline"]


def check_tag_and_inline(design: VibaPool,
                         config: Config = language_config) -> Result:
    """Result[None]: Ok(None) when every product this design writes has all its
    tags different once the inline chains are spread, and every inline chain
    ends; Err names the first mistake, in the order the definitions are
    written.

    ``config`` says which written names stand for the units (VibaReflectConfig);
    the accessor that reads the design is built from it here, so a caller never
    holds one.
    """
    try:
        _Checker(VibaAccess(config)).check(design)
    except (DuplicateTagError, InlineCycleError) as mistake:
        return Err(str(mistake))
    return Ok(None)


class _Checker:
    """Walks the design's descriptors: every product, wherever it sits."""

    def __init__(self, access: VibaAccess):
        self.access = access
        self.visited = set()

    def check(self, design: VibaPool):
        for file in design.files:
            for definition in file.definitions:
                self._check(definition.body)

    def _check(self, descriptor, frame=None):
        """Look at this piece, and at everything written under it.

        ``frame`` is the application this piece is written under, if any: a
        generic's body node is one node, but `Box[int]` and `Box[Pair]` ask two
        different questions of it, so the body is read once per instantiation —
        and once more as written, when the definition itself is walked.
        """
        key = self._key(descriptor, frame)
        if key is not None:
            if key in self.visited:
                return
            self.visited.add(key)
        if descriptor.kind == TYPE_APP:
            self._check_application(descriptor, frame)
            return
        unfolded = self.access.unfold(descriptor)
        if unfolded.kind == TYPE_APP:
            self._check(unfolded, frame)  # a name that landed on an application
            return
        self._dispatch(unfolded, frame)

    def _check_application(self, descriptor, frame):
        """An application: its arguments as written, then the body they make,
        read under this application — that is what tells the instantiations
        apart."""
        for argument in descriptor.payload.args:
            self._check(argument, frame)
        body = self.access.unfold(descriptor)
        if body.kind == TYPE_APP:
            return  # a builtin container, or an application with no body to land on
        self._check(body, self._node_id(descriptor))

    def _dispatch(self, unfolded, frame):
        kind = unfolded.kind
        if kind == PRODUCT:
            self._check_product(unfolded, frame)
        elif kind in (SUM, EXPONENT):
            for element in unfolded.payload.elements:
                self._check(element, frame)
        elif kind == TAGGED:
            self._check(unfolded.payload.tagged_type, frame)
        elif kind == TUPLE:
            for element in unfolded.payload.elements:
                self._check(element, frame)

    def _check_product(self, descriptor, frame):
        """One product: its chain has to end, and its tags — the inlined ones
        included — have to be all different."""
        cycle = self.access.inline_cycle(descriptor)
        if cycle is not None:
            raise InlineCycleError(f"the inline chain comes back to {cycle!r}")
        tags = set()
        for tag, member in self.access.members_of(descriptor) or []:
            if tag is not None:
                if tag in tags:
                    raise DuplicateTagError(
                        f"the tag {tag} is written twice in one product")
                tags.add(tag)
            self._check(member, frame)

    def _key(self, descriptor, frame):
        """What makes this piece the same question: the node it was written as,
        the nodes its arguments were written as (for an application), and the
        application it sits under. `Box[int]` and `Box[Pair]` differ in the
        second; a definition's own body and an instantiation of it differ in
        the third.

        This is also what keeps a design that recurses walkable: `Chain =
        $head int * $tail Chain` (recursion through a tag), `Tree[T] = $leaf T
        * $kids list[Tree[T]]` (a generic that asks for itself) and the
        divergent `W[T] = W[list[T]]` all come round to a piece already looked
        at, and that piece has one answer. A descriptor with no written node (a
        bare unit or zero) has no key and nothing under it to walk.
        """
        node = self._node_id(descriptor)
        if node is None:
            return None
        arguments = ()
        if descriptor.kind == TYPE_APP:
            arguments = tuple(self._node_id(argument) for argument in descriptor.payload.args)
        return (node, arguments, frame)

    def _node_id(self, descriptor):
        node = getattr(getattr(descriptor.payload, "resolvable_type", None), "ast_node", None)
        return None if node is None else id(node)
