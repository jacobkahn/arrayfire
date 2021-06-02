/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <common/half.hpp>
#include <common/jit/UnaryNode.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <af/dim4.hpp>
#include <complex>

namespace cuda {

template<typename To, typename Ti>
struct CastOp {
    const char *name() { return ""; }
};

#define CAST_FN(TYPE)                                \
    template<typename Ti>                            \
    struct CastOp<TYPE, Ti> {                        \
        const char *name() { return "(" #TYPE ")"; } \
    };

CAST_FN(int)
CAST_FN(unsigned int)
CAST_FN(unsigned char)
CAST_FN(unsigned short)
CAST_FN(short)
CAST_FN(float)
CAST_FN(double)

template<typename Ti>
struct CastOp<common::half, Ti> {
    const char *name() { return "(__half)"; }
};

#define CAST_CFN(TYPE)                                    \
    template<typename Ti>                                 \
    struct CastOp<TYPE, Ti> {                             \
        const char *name() { return "__convert_" #TYPE; } \
    };

CAST_CFN(cfloat)
CAST_CFN(cdouble)
CAST_CFN(char)

template<>
struct CastOp<cfloat, cdouble> {
    const char *name() { return "__convert_z2c"; }
};

template<>
struct CastOp<cdouble, cfloat> {
    const char *name() { return "__convert_c2z"; }
};

template<>
struct CastOp<cfloat, cfloat> {
    const char *name() { return "__convert_c2c"; }
};

template<>
struct CastOp<cdouble, cdouble> {
    const char *name() { return "__convert_z2z"; }
};

// Casting from half to unsigned char causes compilation issues. First convert
// to short then to half
template<>
struct CastOp<unsigned char, common::half> {
    const char *name() { return "(short)"; }
};

#undef CAST_FN
#undef CAST_CFN

template<typename To, typename Ti>
struct CastWrapper {
    Array<To> operator()(const Array<Ti> &in) {
        CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        af::dtype to_dtype = static_cast<af::dtype>(dtype_traits<To>::af_type);

        // JIT optimization in the cast of multiple sequential casts that become
        // idempotent - check to see if the previous operation was also a cast
        // TODO: handle arbitrarily long chains of casts
        auto in_node_nary =
            std::dynamic_pointer_cast<common::NaryNode>(in_node);
        if (in_node_nary && in_node_nary->getOp() == af_cast_t) {
            // The only way to get the input type of the child node if it's a
            // cast is to get the output type of the child's child.
            auto in_node_children = in_node_nary->getChildren();
            // Check if any children are casts with the same type - if so,
            // insert a shortcut noop
            for (size_t i = 0; i < in_node_children.size(); ++i) {
                // Found a node whose child can be fast-track noop-ed
                common::Node_ptr in_in_node = in_node_nary->getChildren()[i];
                if (in_in_node->getType() == to_dtype) {
                    // If the output of the child's child is the same as the
                    // output of this node, ignore the input node and simply
                    // connect a noop node from the child's child to produce
                    // this op's output

                    // TODO: including unary.hpp to use
                    // detail::unaryName<af_noop_t> breaks some other stuff
                    return createNodeArray<To>(
                        in.dims(),
                        common::Node_ptr(new common::UnaryNode(
                            to_dtype, "__noop", in_in_node, af_noop_t)));
                }
            }
        }

        common::UnaryNode *node =
            new common::UnaryNode(to_dtype, cop.name(), in_node, af_cast_t);
        return createNodeArray<To>(in.dims(), common::Node_ptr(node));
    }
};

template<typename T>
struct CastWrapper<T, T> {
    Array<T> operator()(const Array<T> &in) { return in; }
};

template<typename To, typename Ti>
Array<To> cast(const Array<Ti> &in) {
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}  // namespace cuda
