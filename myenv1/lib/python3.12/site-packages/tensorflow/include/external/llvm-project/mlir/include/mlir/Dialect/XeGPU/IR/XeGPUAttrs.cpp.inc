/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::xegpu::TensorDescAttr,
::mlir::xegpu::MemoryScopeAttr,
::mlir::xegpu::CachePolicyAttr,
::mlir::xegpu::FenceScopeAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::xegpu::TensorDescAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::TensorDescAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::MemoryScopeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::MemoryScopeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::CachePolicyAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::CachePolicyAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::FenceScopeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::FenceScopeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::xegpu::TensorDescAttr>([&](auto t) {
      printer << ::mlir::xegpu::TensorDescAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::MemoryScopeAttr>([&](auto t) {
      printer << ::mlir::xegpu::MemoryScopeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::CachePolicyAttr>([&](auto t) {
      printer << ::mlir::xegpu::CachePolicyAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::FenceScopeAttr>([&](auto t) {
      printer << ::mlir::xegpu::FenceScopeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace xegpu {
namespace detail {
struct TensorDescAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<MemoryScopeAttr, IntegerAttr, BoolAttr, BoolAttr>;
  TensorDescAttrStorage(MemoryScopeAttr memory_scope, IntegerAttr array_length, BoolAttr boundary_check, BoolAttr scattered) : memory_scope(std::move(memory_scope)), array_length(std::move(array_length)), boundary_check(std::move(boundary_check)), scattered(std::move(scattered)) {}

  KeyTy getAsKey() const {
    return KeyTy(memory_scope, array_length, boundary_check, scattered);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (memory_scope == std::get<0>(tblgenKey)) && (array_length == std::get<1>(tblgenKey)) && (boundary_check == std::get<2>(tblgenKey)) && (scattered == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static TensorDescAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto memory_scope = std::move(std::get<0>(tblgenKey));
    auto array_length = std::move(std::get<1>(tblgenKey));
    auto boundary_check = std::move(std::get<2>(tblgenKey));
    auto scattered = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<TensorDescAttrStorage>()) TensorDescAttrStorage(std::move(memory_scope), std::move(array_length), std::move(boundary_check), std::move(scattered));
  }

  MemoryScopeAttr memory_scope;
  IntegerAttr array_length;
  BoolAttr boundary_check;
  BoolAttr scattered;
};
} // namespace detail
TensorDescAttr TensorDescAttr::get(::mlir::MLIRContext *context, MemoryScopeAttr memory_scope, IntegerAttr array_length, BoolAttr boundary_check, BoolAttr scattered) {
  return Base::get(context, std::move(memory_scope), std::move(array_length), std::move(boundary_check), std::move(scattered));
}

::mlir::Attribute TensorDescAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<MemoryScopeAttr> _result_memory_scope;
  ::mlir::FailureOr<IntegerAttr> _result_array_length;
  ::mlir::FailureOr<BoolAttr> _result_boundary_check;
  ::mlir::FailureOr<BoolAttr> _result_scattered;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_memory_scope = false;
  bool _seen_array_length = false;
  bool _seen_boundary_check = false;
  bool _seen_scattered = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_memory_scope && _paramKey == "memory_scope") {
        _seen_memory_scope = true;

        // Parse variable 'memory_scope'
        _result_memory_scope = ::mlir::FieldParser<MemoryScopeAttr>::parse(odsParser);
        if (::mlir::failed(_result_memory_scope)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_TensorDescAttr parameter 'memory_scope' which is to be a `MemoryScopeAttr`");
          return {};
        }
      } else if (!_seen_array_length && _paramKey == "array_length") {
        _seen_array_length = true;

        // Parse variable 'array_length'
        _result_array_length = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_array_length)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_TensorDescAttr parameter 'array_length' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_boundary_check && _paramKey == "boundary_check") {
        _seen_boundary_check = true;

        // Parse variable 'boundary_check'
        _result_boundary_check = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_boundary_check)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_TensorDescAttr parameter 'boundary_check' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_scattered && _paramKey == "scattered") {
        _seen_scattered = true;

        // Parse variable 'scattered'
        _result_scattered = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_scattered)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_TensorDescAttr parameter 'scattered' which is to be a `BoolAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return TensorDescAttr::get(odsParser.getContext(),
      MemoryScopeAttr((_result_memory_scope.value_or(MemoryScopeAttr()))),
      IntegerAttr((_result_array_length.value_or(IntegerAttr()))),
      BoolAttr((_result_boundary_check.value_or(BoolAttr()))),
      BoolAttr((_result_scattered.value_or(BoolAttr()))));
}

void TensorDescAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getMemoryScope() == MemoryScopeAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "memory_scope = ";
      if (!(getMemoryScope() == MemoryScopeAttr())) {
        odsPrinter.printStrippedAttrOrType(getMemoryScope());
      }
    }
    if (!(getArrayLength() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "array_length = ";
      if (!(getArrayLength() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getArrayLength());
      }
    }
    if (!(getBoundaryCheck() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "boundary_check = ";
      if (!(getBoundaryCheck() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getBoundaryCheck());
      }
    }
    if (!(getScattered() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "scattered = ";
      if (!(getScattered() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getScattered());
      }
    }
  }
  odsPrinter << ">";
}

MemoryScopeAttr TensorDescAttr::getMemoryScope() const {
  return getImpl()->memory_scope;
}

IntegerAttr TensorDescAttr::getArrayLength() const {
  return getImpl()->array_length;
}

BoolAttr TensorDescAttr::getBoundaryCheck() const {
  return getImpl()->boundary_check;
}

BoolAttr TensorDescAttr::getScattered() const {
  return getImpl()->scattered;
}

} // namespace xegpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::TensorDescAttr)
namespace mlir {
namespace xegpu {
namespace detail {
struct MemoryScopeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::MemoryScope>;
  MemoryScopeAttrStorage(::mlir::xegpu::MemoryScope value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MemoryScopeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MemoryScopeAttrStorage>()) MemoryScopeAttrStorage(std::move(value));
  }

  ::mlir::xegpu::MemoryScope value;
};
} // namespace detail
MemoryScopeAttr MemoryScopeAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::MemoryScope value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute MemoryScopeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::MemoryScope> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::MemoryScope> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeMemoryScope(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::MemoryScope" << " to be one of: " << "global" << ", " << "slm")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_MemoryScopeAttr parameter 'value' which is to be a `::mlir::xegpu::MemoryScope`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MemoryScopeAttr::get(odsParser.getContext(),
      ::mlir::xegpu::MemoryScope((*_result_value)));
}

void MemoryScopeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMemoryScope(getValue());
}

::mlir::xegpu::MemoryScope MemoryScopeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace xegpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemoryScopeAttr)
namespace mlir {
namespace xegpu {
namespace detail {
struct CachePolicyAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::CachePolicy>;
  CachePolicyAttrStorage(::mlir::xegpu::CachePolicy value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CachePolicyAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CachePolicyAttrStorage>()) CachePolicyAttrStorage(std::move(value));
  }

  ::mlir::xegpu::CachePolicy value;
};
} // namespace detail
CachePolicyAttr CachePolicyAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::CachePolicy value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CachePolicyAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::CachePolicy> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::CachePolicy> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeCachePolicy(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::CachePolicy" << " to be one of: " << "cached" << ", " << "uncached" << ", " << "streaming" << ", " << "read_invalidate" << ", " << "write_back" << ", " << "write_through")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_CacheHintAttr parameter 'value' which is to be a `::mlir::xegpu::CachePolicy`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CachePolicyAttr::get(odsParser.getContext(),
      ::mlir::xegpu::CachePolicy((*_result_value)));
}

void CachePolicyAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyCachePolicy(getValue());
  odsPrinter << ">";
}

::mlir::xegpu::CachePolicy CachePolicyAttr::getValue() const {
  return getImpl()->value;
}

} // namespace xegpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::CachePolicyAttr)
namespace mlir {
namespace xegpu {
namespace detail {
struct FenceScopeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::FenceScope>;
  FenceScopeAttrStorage(::mlir::xegpu::FenceScope value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FenceScopeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FenceScopeAttrStorage>()) FenceScopeAttrStorage(std::move(value));
  }

  ::mlir::xegpu::FenceScope value;
};
} // namespace detail
FenceScopeAttr FenceScopeAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::FenceScope value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute FenceScopeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::FenceScope> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::FenceScope> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeFenceScope(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::FenceScope" << " to be one of: " << "workgroup" << ", " << "gpu")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_FenceScopeAttr parameter 'value' which is to be a `::mlir::xegpu::FenceScope`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FenceScopeAttr::get(odsParser.getContext(),
      ::mlir::xegpu::FenceScope((*_result_value)));
}

void FenceScopeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyFenceScope(getValue());
}

::mlir::xegpu::FenceScope FenceScopeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace xegpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::FenceScopeAttr)
namespace mlir {
namespace xegpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute XeGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void XeGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace xegpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

